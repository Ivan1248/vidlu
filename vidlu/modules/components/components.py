from vidlu.utils.func import partial
from functools import partialmethod
from collections.abc import Sequence
import typing as T
import math
import numpy as np
import warnings

import torch
from torch import nn
from torch.nn import functional as F
import einops

import vidlu.modules.elements as E
from vidlu.modules import tensor_extra
from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj
from vidlu.utils.func import params, Reserved, default_args, Empty, ArgTree, tree_partial
from vidlu.utils.collections import NameDict
import vidlu.modules.utils as vmu

from . import _default_factories as D


# Constant functions


class GaussianFilter2D(E.Module):
    def __init__(self, sigma=2, ksize=None, sigma_to_ksize_ratio=4, padding_mode='reflect'):
        if ksize is None:
            ksize = int(sigma_to_ksize_ratio * sigma)
            ksize += int(ksize % 2 == 0)
        elif ksize % 2 == 0:
            raise ValueError("`ksize` is required to be odd.")
        super().__init__()
        self.padding = [ksize // 2] * 4
        self.padding_mode = padding_mode

        with torch.no_grad():
            g = torch.arange(ksize, dtype=torch.float).sub_((ksize - 1) / 2)
            kernel = torch.exp(-g.pow_(2).div_(2 * sigma ** 2))
            self.register_buffer('kernel', kernel.div_(torch.sum(kernel)))  # normalize
            self.kernel.requires_grad_(False)

    def forward(self, x):
        ker1 = self.kernel.expand(x.shape[1], 1, 1, *self.kernel.shape)
        ker2 = ker1.view(x.shape[1], 1, *self.kernel.shape, 1)
        x = F.pad(x, self.padding, mode=self.padding_mode)
        for ker in [ker1, ker2]:
            x = F.conv2d(x, weight=ker, groups=x.shape[1], padding=0)
        return x


class SquareFilter(E.Module):
    def __init__(self, ksize=None, padding_mode='reflect'):
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        super().__init__()
        self.padding = [ksize // 2] * 4
        self.padding_mode = padding_mode

        with torch.no_grad():
            kernel = torch.ones(ksize)
            self.register_buffer('kernel', kernel.div_(torch.sum(kernel)))  # normalize
            self.kernel.requires_grad_(False)

    def forward(self, x):
        ker1 = self.kernel.expand(x.shape[1], 1, 1, *self.kernel.shape)
        ker2 = ker1.view(x.shape[1], 1, *self.kernel.shape, 1)
        x = F.pad(x, self.padding, mode=self.padding_mode)
        for ker in [ker1, ker2]:
            x = F.conv2d(x, weight=ker, groups=x.shape[1], padding=0)
        return x


# Activations ######################################################################################


class Tent(E.Module):
    def __init__(self, channelwise=False, delta_range=(0.05, 1.)):
        super().__init__()
        self.channelwise = channelwise
        self.min_delta, self.max_delta = delta_range
        self.delta = None

    def build(self, x):
        self.delta = nn.Parameter(
            torch.ones(x.shape[1]) if self.channelwise else torch.tensor(self.max_delta),
            requires_grad=True)

    def forward(self, x):
        with torch.no_grad():
            self.delta.clamp_(self.min_delta, self.max_delta)
        delta = self.delta.view(list(self.delta.shape) + [1] * (len(x.shape) - 2))
        # return F.relu(delta - (x - delta).abs())  # centered at delta
        return F.relu(delta - x.abs())


# Transformer attention ############################################################################

class MultiHeadAttention(E.Module):
    # Modified from https://einops.rocks/pytorch-examples.html
    # PyTorch provides a more efficient implementation.
    def __init__(self, num_heads, *, embed_dim=None, d_k, d_v):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim, self.d_k, self.d_v = embed_dim, d_k, d_v

        self.w_qs, self.w_ks, self.w_vs = [nn.Linear(dim, dim) for dim in [d_k, d_k, d_v]]
        self.fc_o = nn.Linear(num_heads * d_v, embed_dim)

        if embed_dim is not None:
            self.reset_parameters()

    def build(self, q, k, v):
        if self.embed_dim is None:
            self.embed_dim = q.shape[1]
            self.reset_parameters()

    def reset_parameters(self):
        for w, dim in zip([self.w_qs, self.w_ks, self.w_vs], [self.d_k, self.d_k, self.d_v]):
            nn.init.normal_(w.weight, mean=0, std=np.sqrt(2.0 / (self.embed_dim + dim)))
        nn.init.xavier_normal_(self.fc_o.weight)

    def forward(self, q, k, v, mask=None):
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.num_heads)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.num_heads)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.num_heads)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.fc_o(output)
        return output, attn


# Blocks ###########################################################################################


def _add_norm_act(seq, suffix, norm_f, act_f):
    if norm_f:
        seq.add(f'norm{suffix}', norm_f())
    seq.add(f'act{suffix}', act_f())


def _resolve_block_args(kernel_sizes, base_width, width_factors, stride, dilation,
                        stride_after_1x1):
    if not len(kernel_sizes) == len(width_factors):
        raise ValueError(f"{len(kernel_sizes)=} does not match {len(width_factors)=}.")
    widths = [base_width * wf for wf in width_factors]
    round_widths = list(map(round, widths))  # fractions to ints
    if widths != round_widths:
        raise ValueError(f"Some widths are not integers (width_factors={width_factors},"
                         f" base_width={base_width}; widths={widths}).")
    widths = round_widths
    conv_defaults = default_args(D.conv_f)
    if not isinstance(stride, Sequence):
        if stride_after_1x1 and kernel_sizes[0] == 1 and stride > 1:
            # if the first convolution is 1x1, it is not strided -- see the Torchvision Bottleneck documentation
            stride = [1, stride] + [1] * (len(widths) - 2)
        else:
            stride = [stride] + [1] * (len(widths) - 1)
    if not isinstance(dilation, Sequence):
        dilation = [dilation] + [conv_defaults.dilation] * (len(widths) - 1)
    return widths, stride, dilation


class PreactBlock(E.Seq):
    """A block of one or more sequences of the form
    "normalization -> activation [-> noise] -> convolution".

    Args:
        kernel_sizes (Sequence[int]): Kernel sizes of convolutions.
        base_width (int): Base width that is multiplied by a width factor from
            `width_factors` to get its output widths for all convolutions.
        width_factors (Sequence[int]): Factors that multiply `base_width` to get
            the output width for all convolutions.
        stride (int or Sequence[int]): Stride of the first convolution or a
            sequence of strides of all convolutions.
        dilation (int or Sequence[int]): Dilation of the first convolution or a
            sequence of dilations of all convolutions.
        noise_locations (Sequence[int]): The indices of convolutions that are to
            be followed by a noise layer.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
        conv_f: Convolution module factory.
        noise_f: Noise module factory.
    """

    def __init__(self,
                 *,
                 kernel_sizes,
                 base_width,
                 width_factors,
                 stride=1,
                 dilation=1,
                 noise_locations=(),
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 conv_f=partial(D.conv_f,
                                kernel_size=Reserved,
                                out_channels=Reserved,
                                stride=Reserved,
                                dilation=Reserved),
                 noise_f=None,
                 stride_after_1x1=True):
        super().__init__()
        widths, stride, dilation = _resolve_block_args(kernel_sizes, base_width, width_factors,
                                                       stride, dilation, stride_after_1x1)
        for i, (k, w, s, d) in enumerate(zip(kernel_sizes, widths, stride, dilation)):
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add(f'noise{i}', noise_f())
            self.add(f'conv{i}',
                     Reserved.call(conv_f, out_channels=w, kernel_size=k, stride=s, dilation=d))


class PostactBlock(E.Seq):
    """A block of one or more sequences of the form
    "convolution -> normalization -> activation [-> noise]".

    Args:
        kernel_sizes (Sequence[int]): Kernel sizes of convolutions.
        base_width (int): Base width that is multiplied by a width factor from
            `width_factors` to get its output widths for all convolutions.
        width_factors (Sequence[int]): Factors that multiply `base_width` to get
            the output width for all convolutions.
        noise_locations (Sequence[int]): The indices of activations that are to
            be followed by a noise layer.
        stride (int or Sequence[int]): Stride of the first convolution or a
            sequence of strides of all convolutions.
        dilation (int or Sequence[int]): Dilation of the first convolution or a
            sequence of dilations of all convolutions.
        conv_f: Convolution module factory.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
        noise_f: Noise module factory.
    """

    def __init__(self,
                 *,
                 kernel_sizes,
                 base_width,
                 width_factors,
                 stride=1,
                 dilation=1,
                 noise_locations=(),
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 conv_f=params(PreactBlock).conv_f,
                 noise_f=None,
                 stride_after_1x1=True):
        super().__init__()
        widths, stride, dilation = _resolve_block_args(kernel_sizes, base_width, width_factors,
                                                       stride, dilation, stride_after_1x1)
        for i, (k, w, s, d) in enumerate(zip(kernel_sizes, widths, stride, dilation)):
            self.add(f'conv{i}',
                     Reserved.call(conv_f, out_channels=w, kernel_size=k, stride=s, dilation=d))
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add(f'noise{i}', noise_f())


def _check_block_args(block_f):
    args = params(block_f)
    if 'kernel_sizes' in args and args['kernel_sizes'] is Empty:
        raise ValueError("Argument kernel_sizes missing in block_f.")


# Stem blocks ######################################################################################


class StandardRootBlock(E.Seq):
    """Standard ResNet/DenseNet root block.

    Args:
        out_channels (int): number of output channels.
        small_input (bool): If True, the root block doesn't reduce spatial dimensions. E.g. it
            should be `True` for CIFAR-10, but `False` for ImageNet.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
    """

    def __init__(self,
                 out_channels: int,
                 small_input,
                 conv_f=partial(E.Conv,
                                kernel_size=Reserved,
                                stride=Reserved,
                                padding='half',
                                bias=Reserved),
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 pool_f=E.MaxPool):
        conv_args = dict(out_channels=out_channels, dilation=1)
        if small_input:  # CIFAR
            super().__init__(
                conv=Reserved.call(conv_f, **conv_args, kernel_size=3, stride=1, bias=True))
        else:
            super().__init__(conv=Reserved.call(
                conv_f, **conv_args, kernel_size=7, stride=2, bias=norm_f is None))
            if norm_f is not None:
                self.add(norm=norm_f())
            self.add(act=act_f(),
                     pool=pool_f(3, stride=2, padding='half'))


class ImprovedRootBlock(E.Seq):
    """Standard Inception-v2 root block.

    This is also used in ResNet-C, SENet, PSPNet, DeepLabV3 and ShuffleNetV2.

    Args:
        out_channels (int): number of output channels.
    """

    def __init__(self,
                 out_channels: int,
                 init_channels=32,
                 block_f=partial(PreactBlock,
                                 kernel_sizes=[3, 3, 3],
                                 stride=[2, 2, 1],
                                 width_factors=Reserved),
                 pool_f=E.MaxPool):
        super().__init__(
            block=Reserved.call(block_f,
                                base_width=out_channels,
                                width_factors=[min(1, init_channels / out_channels)] * 2 + [1]),
            pool=pool_f(3, stride=2, padding='half'))


# Bijective or injective ###########################################################################


class AffineCoupling(E.Module):
    # http://arxiv.org/abs/1605.08803 (Real NVP)
    def __init__(self, scale_f, translation_f):
        super().__init__()
        self.scale, self.translation = scale_f(), translation_f()

    def forward(self, x1, x2):
        s = self.scale(x1).exp()
        y1 = Ladj.stop(x1)
        y2 = x2 * s + self.translation(x1)
        return Ladj.add((y1, y2), (x1, x2), lambda: s.view(s.shape[0], -1).sum(1))

    def inverse_forward(self, y1, y2):
        if Ladj.has(y1):
            raise NotImplementedError()
        return y1, (y2 - self.translation(y1)) * (-self.scale(y1)).exp()


@E.zero_log_abs_det_jac
class AdditiveCoupling(E.Module):
    # http://arxiv.org/abs/1410.8516 (NICE)
    def __init__(self, translation_f):
        super().__init__()
        self.translation = translation_f()

    def forward(self, x1, x2):
        return x1, x2 + self.translation(x1)

    def inverse_forward(self, y1, y2):
        return y1, y2 - self.translation(y1)


@E.zero_log_abs_det_jac
class ProportionSplit(E.Module):
    reconstruction_tols = dict(rtol=0, atol=0)

    def __init__(self, proportion: float, dim=1, rounding=round):
        super().__init__()
        self.proportion, self.dim, self.round = proportion, dim, rounding

    def forward(self, x):
        m = x.shape[self.dim]
        n = self.round(self.proportion * m)
        return x.split([n, m - n], dim=self.dim)

    def inverse_module(self):
        return E.Concat(dim=self.dim)


class _PadChannelsBase(E.Module):
    reconstruction_tols = dict(rtol=0, atol=0)

    def __init__(self, padding):
        if not isinstance(padding, int) and len(padding) != 2:
            raise ValueError("`padding` should be a single `int` or a sequence of length 2.")
        super().__init__()
        self.padding = [0, padding] if isinstance(padding, int) else list(padding)


def pad_channels(x, padding):
    y = F.pad(x, [0, 0, 0, 0] + padding)
    if Ladj.has(x) and y.shape != x.shape:
        # raise RuntimeError("Non-zero padding does not have a square jacobian.")
        warnings.warn("Non-null padding does not have a square jacobian.")
    return Ladj.add(y, x, lambda: 0)


class PadChannels(_PadChannelsBase):
    def forward(self, x):  # injective
        return pad_channels(x, self.padding)

    def inverse_module(self):
        return UnpadChannels(self.padding)


class UnpadChannels(_PadChannelsBase):
    def forward(self, x):
        return x[:, self.padding[0]:x.shape[1] - self.padding[1], :, :]

    def inverse_module(self):
        return PadChannels(self.padding)


class Fold(E.Module):
    def __init__(self, block_size):
        self.block_size = b = block_size

    def forward(self, x):
        return F.fold()


class Baguette(E.Seq):
    """Rearranges a NCHW array into a N(C*b*b)(H/b)(W/b) array.

    From
    https://github.com/jhjacobsen/pytorch-i-revnet/issues/18#issue-486527382"""
    reconstruction_tols = dict(rtol=0, atol=0)

    def __init__(self, block_size, variant=0):
        self.block_size = b = block_size
        if b == 1:
            super().__init__()
        else:
            if variant == 0:
                super().__init__(
                    resh1=E.BatchReshape(lambda c, h, w: (c, h // b, b, w // b, b)),
                    perm1=E.Permute(0, 3, 5, 1, 2, 4),  # n c h/b bh w/b bw -> n bh bw c h/b w/b
                    resh2=E.BatchReshape(lambda bh, bw, c, h_b, w_b: (bh * bw * c, h_b, w_b)))
            # TODO: try alternatives: unfold, nn.PixelShuffle, or the code below
            # https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
            # https://stackoverflow.com/questions/58857720/is-there-an-equivalent-pytorch-function-for-tf-nn-space-to-depth
            else:
                super().__init__(
                    resh1=E.BatchReshape(lambda c, h, w: (b, b, c // b ** 2, h, w)),
                    perm1=E.Permute(0, 3, 4, 1, 5, 2),
                    # n b_1 b_2 c/b**2 h w -> n c/b**2 h b_1 w b_2
                    resh2=E.BatchReshape(lambda c_b2, h, b_1, w, b_2: (c_b2, h * b_1, w * b_2)))

    def __repr__(self):
        return f"{type(self).__name__}({self.block_size})"

    def __str__(self):
        return repr(self)

    def __getstate__(self):
        return self.block_size

    def __setstate__(self, state):
        self.__dict__.clear()
        self.__init__(state)

    # TODO: IMPROVING GLOW https://arogozhnikov.github.io/einops/pytorch-examples.html


class Invertible1x1Conv(E.Conv):  # TODO: implement QR parametrization, which can be a little faster
    def __init__(self, num_channels):
        super().__init__(self, num_channels, 1, bias=False)

    @torch.no_grad()
    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = torch.qr(torch.randn((self.num_channels, self.num_channels))).Q
        self.weight[:] = w_init[:, :, None, None]

    def forward(self, x):
        y = super().forward(x)
        return Ladj.add(y, x, lambda: self._ladj(y, x))

    def inverse_forward(self, y):
        w_inv = torch.inverse(self.weight.squeeze())[:, :, None, None]
        x = F.conv2d(y, w_inv, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return Ladj.add(x, y, lambda: self._ladj(x, y, inverse=True))

    def _ladj(self, y, x, inverse=False):
        a, ladj_w = np.prod(x.shape[2:]), self.orig.weight.squeeze().det().abs().log()
        return (-a if inverse else a) * ladj_w


class ChannelMeanSplitter(E.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        self.keep_dims = keepdim

    def forward(self, x):
        mean = x.view(*x.shape[:2], -1).mean(-1)
        if self.keep_dims:
            mean = self._align_dims(mean, x)
        y = (mean, x - self._align_dims(mean, x))
        return Ladj.add(y, x, Ladj.zero(x))

    def inverse_forward(self, mean_other):
        mean, other = mean_other
        x = self._align_dims(mean, other) + other
        return Ladj.add(x, mean_other, Ladj.zero(x))

    def _align_dims(self, mean, other):
        broadcast_shape = list(mean.shape) + [1] * (other.dim() - mean.dim())
        return mean.view(broadcast_shape)


# Attention ########################################################################################


class SqueezeExcitation(E.Module):
    def __init__(self, channel, reduction=16, squeeze_f=nn.AdaptiveAvgPool2d, act_f=E.ReLU):
        super().__init__()
        self.args = self.get_args(locals())

    def build(self, x):
        a = self.args
        self.attention = E.Seq(squeeze=a.squeeze_f(output_size=1),
                               lin0=E.Linear(x.shape[1] // a.reduction, bias=False),
                               act0=a.act_f(),
                               lin1=E.Linear(x.shape[1], bias=False),
                               act1=nn.Sigmoid(),
                               resh=E.BatchReshape(x.shape[1], 1, 1))

    def forward(self, x):
        return x * self.attention(x)


# Resizing #########################################################################################


class Resize(E.Module):
    """Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`. See :func:`~torch.nn.F.interpolate` for details.

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial shape.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has
            to match input size if it is a tuple.
        mode (string): algorithm used for upsampling: 'nearest' | 'linear'
            | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.args = self.get_args(locals())

    def forward(self, x):
        a = self.args
        return F.interpolate(x,
                             size=a.size,
                             scale_factor=a.scale_factor,
                             mode=a.mode,
                             align_corners=a.align_corners)


class MultiScaleEnsemble(E.Module):
    def __init__(self, module_f, scales=(1,), weights=None, interp_mode='bilinear',
                 align_corners=False):
        super().__init__()
        self.args = self.get_args(locals())
        self.module = module_f()

    def forward(self, x):
        a = self.args
        xs = [x if s == 1 else
              F.interpolate(x, scale_factor=s, mode=a.interp_mode, align_corners=a.align_corners)
              for s in a.scales]
        ys = tuple(map(self.module, xs))
        del xs
        shape = ys[0].shape[-2:]
        ys = [ys[0]] + [
            F.interpolate(y, size=shape, mode=a.interp_mode, align_corners=a.align_corners)
            for y in ys[1:]]
        # for x, y in zip(xs, ys):
        #     from vidlu.utils.presentation import show_segmentation_prediction
        #     show_segmentation_prediction(y, x)
        if a.weights is None:
            y = ys[0]
            for z in ys:
                y += z
            y.div_(len(ys))
        else:
            y = ys[0].mul_(a.weights[0]) if a.weights[0] != 1 else ys[0]
            for z, w in zip(ys[1:], a.weights[1:]):
                y.add_(z, alpha=w) if w != 1 else y.add_(z)
        return y


# Pyramid pooling ##################################################################################


class DenseSPP(E.Module):
    # Spatial pyramid pooling for dense prediction
    def __init__(self,
                 bottleneck_size=512,
                 level_size=128,
                 out_size=128,
                 grid_sizes=(8, 4, 2, 1),
                 block_f=partial(PreactBlock,
                                 base_width=Reserved,
                                 kernel_sizes=[1],
                                 width_factors=[1]),
                 upsample=partial(F.interpolate, mode='bilinear', align_corners=False),
                 square_output=False):
        super().__init__()
        self.grid_sizes = grid_sizes
        self.upsample = upsample
        self.input_block = block_f(base_width=bottleneck_size)  # reduces the number of channels
        self.pyramid = E.ModuleTable(
            {f'block{i}': block_f(base_width=level_size)
             for i in range(len(grid_sizes))})
        self.fuse_block = block_f(base_width=out_size)
        self.square_output = square_output

    def forward(self, x):
        target_size = x.size()[2:4]
        if not self.square_output:
            ar = target_size[1] / target_size[0]
            if ar == 0:
                warnings.warn("SPP input height is greater than its width.")

        x = self.input_block(x)
        levels = [x]
        for pyr_block, grid_size in zip(self.pyramid, self.grid_sizes):
            if not self.square_output:  # keep aspect ratio
                grid_size = (grid_size, max(1, round(ar * grid_size)))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)  # TODO: x vs levels[-1]
            level = pyr_block(x_pooled)
            levels.append(self.upsample(level, target_size))
        return self.fuse_block(torch.cat(levels, 1))


# Ladder Upsampling ################################################################################


class LadderUpsampleBlend(E.Module):
    """For LadderDenseNet and SwiftNet. """
    _pre_blendings = dict(concat=E.Concat, sum=E.Sum)

    def __init__(self,
                 out_channels: int,
                 pre_blending: T.Literal['concat', 'sum'] = 'concat',
                 blend_block_f: T.Callable[[], nn.Module] = partial(
                     PreactBlock, base_width=Reserved, width_factors=Reserved),
                 non_multiple_effect: T.Literal[None, 'warn', 'error'] = 'warn'):
        super().__init__()
        if pre_blending not in self._pre_blendings:
            raise ValueError(f"Invalid pre-blending '{pre_blending}'. " +
                             f"It should be one of {tuple(self._pre_blendings.keys())}.")
        self.block_f = Reserved.partial(blend_block_f, width_factors=[1])

        self.project = None
        self.pre_blend = self._pre_blendings[pre_blending]()
        self.blend = self.block_f(base_width=out_channels, kernel_sizes=[3])
        self.non_multiple_effect = non_multiple_effect

    def build(self, x, skip):
        self.project = Reserved.partial(self.block_f, base_width=x.shape[1])(kernel_sizes=[1])

    def _check_sizes(self, x, skip):
        if self.non_multiple_effect is not None:
            x_size, skip_size = tuple(x.shape[-2:]), tuple(skip.shape[-2:])
            rem = np.mod(skip_size, x_size).max()
            factor = np.array(skip_size) / np.array(x_size)
            if factor[0] != factor[1] or rem != 0:
                message = f"The higher-resolution input size ({tuple(skip_size)}) is not a" \
                          + f" multiple of lower-resolution input size ({tuple(x_size)})."
                if self.non_multiple_effect == 'warn':
                    warnings.warn(message)
                else:
                    raise RuntimeError(message)

    def forward(self, x, skip):
        self._check_sizes(x, skip)

        x_up = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip_proj = self.project(skip)
        b = self.pre_blend((x_up, skip_proj))
        return self.blend(b)


class KresoLadder(E.Module):
    def __init__(self, width, up_blend_f=LadderUpsampleBlend):
        super().__init__()
        self.width, self.up_blend_f = width, up_blend_f
        self.up_blends = None

    def _build(self, stage_count):
        self.up_blends = nn.ModuleList([self.up_blend_f(self.width) for _ in range(stage_count)])

    def build(self, x, skips):
        self._build(len(skips))

    def forward(self, x, skips):
        ups = [x]
        for upbl, skip in zip(self.up_blends, skips):
            ups.append(upbl(ups[-1], skip))  # TODO: remove detach
        return ups[-1]


class KresoContext(E.Seq):
    def __init__(self,
                 base_width=128,
                 block_f=partial(PreactBlock,
                                 base_width=Reserved,
                                 kernel_sizes=Reserved,
                                 width_factors=Reserved,
                                 dilation=Reserved)):
        super().__init__(context=Reserved.call(block_f,
                                               base_width=base_width,
                                               kernel_sizes=[1, 3],
                                               width_factors=[4, 1],
                                               dilation=[1, 2]))


class KresoLadderDecoder(E.Module):
    def __init__(self,
                 up_width: int,
                 context_f=DenseSPP,
                 up_blend_f=LadderUpsampleBlend,
                 post_activation=False,
                 lateral_preprocessing=lambda x: x):
        super().__init__()
        self.context = context_f()
        self.ladder = KresoLadder(up_width, up_blend_f)  # initialized when called
        self.post_activation = post_activation
        if post_activation:
            defaults = default_args(default_args(up_blend_f).blend_block_f)
            self.norm, self.act = defaults.norm_f(), defaults.act_f()
        self.lateral_preprocessing = lateral_preprocessing

    def forward(self, context_input, laterals):
        context = self.context(context_input)
        laterals = list(map(self.lateral_preprocessing, laterals))
        ladder_output = self.ladder(context, laterals[::-1])
        return self.act(self.norm(ladder_output)) if self.post_activation else ladder_output


class LadderModel(nn.Module):
    def __init__(self, backbone_f, laterals: T.Sequence[str], decoder_f):
        self.backbone = backbone_f()
        self.laterals = laterals
        self.decoder = decoder_f()

    def forward(self, x):
        context_input, laterals = E.with_intermediate_outputs(self.backbone, self.laterals)(x)
        return self.decoder(context_input, laterals)


class KresoLadderModel(KresoLadderDecoder):
    def __init__(self,
                 backbone_f,
                 laterals: T.Sequence[str],
                 up_width: int,
                 context_f=DenseSPP,
                 up_blend_f=LadderUpsampleBlend,
                 post_activation=False,
                 lateral_preprocessing=lambda x: x):
        super().__init__(
            up_width=up_width, context_f=context_f,
            up_blend_f=up_blend_f, post_activation=post_activation,
            lateral_preprocessing=lateral_preprocessing)
        self.backbone = backbone_f()
        self.laterals = laterals

    def forward(self, x):
        context_input, laterals = E.with_intermediate_outputs(self.backbone, self.laterals)(x)
        return super().forward(context_input, laterals)


class MorsicPyramid(E.Module):
    def __init__(self, model, stage_count: int, laterals: T.Sequence[str], width,
                 sum_f=E.Sum):
        super().__init__()
        self.model = model
        self.stage_count = stage_count  # number of pyramid stages
        self.laterals = laterals
        self.projs = nn.ModuleList(
            [nn.ModuleList([E.Conv(kernel_size=1, out_channels=width, bias=True)
                            for _ in range(len(laterals) + 1)])
             for _ in range(self.stage_count)])
        self.sum = sum_f()

    def forward(self, x):
        stage_outputses = [[None] * (len(self.laterals) + self.stage_count - 1)
                           for _ in range(self.stage_count)]
        for i, projs in enumerate(self.projs):
            xi = F.interpolate(x, scale_factor=0.5 ** i, mode='bilinear',
                               align_corners=False) if i != 0 else x
            final, laterals = E.with_intermediate_outputs(self.model, self.laterals)(xi)
            outputs = [proj(out) for (out, proj) in zip([*laterals, final], projs)]
            stage_outputses[i][i:i + len(outputs)] = outputs
        column_sums = [self.sum([xi for xi in col if xi is not None])
                       for col in zip(*stage_outputses)]
        return column_sums


class MorsicPyramidModel(E.Module):
    def __init__(self,
                 backbone_f,
                 laterals: T.Sequence[str],
                 stage_count: int,
                 up_width: int,
                 up_blend_f=LadderUpsampleBlend,
                 post_activation=False):
        super().__init__()
        self.backbone = backbone_f()
        self.ladder = KresoLadder(up_width, up_blend_f)  # initialized when called
        self.pyr = MorsicPyramid(self.backbone, stage_count=stage_count, laterals=laterals,
                                 width=up_width)
        self.post_activation = post_activation
        if post_activation:
            defaults = default_args(default_args(up_blend_f).blend_block_f)
            self.norm, self.act = defaults.norm_f(), defaults.act_f()

    def forward(self, x):
        pyr_out = self.pyr(x)
        ladder_output = self.ladder(pyr_out[-1], pyr_out[:-1][::-1])
        return self.act(self.norm(ladder_output)) if self.post_activation else ladder_output


# ResNetV1 #########################################################################################


def _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, conv_f, norm_f):
    if stride == 1 and in_width == out_width:
        return nn.Identity()
    if dim_change in ('proj', 'pool_proj', 'conv3'):
        shortcut = E.Seq()
        if dim_change == 'pool_proj' and stride == 2:  # ResNet-D
            shortcut.add(pool_f=E.AvgPool(2, stride=2, ))
        shortcut.add(conv=conv_f(out_channels=out_width,
                                 kernel_size=3 if dim_change == 'conv3' else 1,
                                 stride=1 if dim_change == 'pool_proj' else stride,
                                 padding='half',
                                 dilation=1,
                                 bias=False))
        if norm_f is not None:
            shortcut.add(norm=norm_f())
        return shortcut
    elif dim_change == 'pad':
        pad = [0] * 5 + [out_width - in_width]
        return E.Seq(pool=E.AvgPool(stride, stride),
                     pad=E.Func(partial(F.pad, pad=pad, mode='constant')))


def _check_resnet_unit_args(block_f, dim_change):
    if dim_change not in ['pad', 'proj', 'conv3']:
        raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")
    _check_block_args(block_f)


class ResNetV1Unit(E.Seq):
    def __init__(self, block_f=partial(PostactBlock, stride_after_1x1=True), dim_change='proj',
                 inplace_add=False):
        _check_resnet_unit_args(block_f, dim_change)
        super().__init__()
        self.block_f, self.dim_change, self.inplace_add = block_f, dim_change, inplace_add

    def build(self, x):
        block_args = default_args(self.block_f)
        shortcut = _get_resnetv1_shortcut(in_width=x.shape[1],
                                          out_width=block_args.base_width *
                                                    block_args.width_factors[-1],
                                          stride=block_args.stride,
                                          dim_change=self.dim_change,
                                          conv_f=block_args.conv_f,
                                          norm_f=block_args.norm_f)
        block = Reserved.call(self.block_f)[:-1]  # block without the last activation
        self.add(fork=E.Fork(block=block, shortcut=shortcut),
                 sum=E.Sum(inplace=self.inplace_add),
                 act=default_args(self.block_f).act_f())


class ResNetV1Groups(E.Seq):
    def __init__(self,
                 group_lengths,
                 base_width,
                 block_f=partial(default_args(ResNetV1Unit).block_f,
                                 base_width=Reserved,
                                 width_factors=(2, 2),
                                 stride=Reserved),
                 dim_change=default_args(ResNetV1Unit).dim_change,
                 unit_f=ResNetV1Unit):
        super().__init__()
        _check_block_args(block_f)
        for i, l in enumerate(group_lengths):
            for j in range(l):
                u = unit_f(block_f=Reserved.partial(block_f,
                                                    base_width=base_width * 2 ** i,
                                                    stride=1 + int(i > 0 and j == 0)),
                           dim_change=dim_change)
                self.add(f'unit{i}_{j}', u)


class ResNetV1Backbone(E.Seq):
    """Resnet (V1) backbone.

    Args:
        base_width (int): number of channels in the output of the root block and the base width of
            blocks in the first group.
        small_input (bool): If True, the root block doesn't reduce spatial dimensions. E.g. it
            should be `True` for CIFAR-10, but `False` for ImageNet.
        group_lengths:
        width_factors:
        block_f:
        dim_change:
    """

    def __init__(self,
                 small_input=False,
                 base_width=64,
                 group_lengths=(2,) * 4,
                 block_f=default_args(ResNetV1Groups).block_f,
                 dim_change=default_args(ResNetV1Groups).dim_change,
                 groups_f=ResNetV1Groups):
        _check_block_args(block_f)
        block_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f', 'conv_f']}
        super().__init__(root=StandardRootBlock(base_width, small_input, **block_args),
                         bulk=groups_f(group_lengths,
                                       base_width=base_width,
                                       block_f=block_f,
                                       dim_change=dim_change))


# ResNetV2 #########################################################################################


def _get_resnetv2_shortcut(in_width, out_width, stride, dim_change, conv_f):
    if stride == 1 and in_width == out_width:
        return nn.Identity()
    else:
        if dim_change in ('proj', 'conv3'):
            return conv_f(out_channels=out_width,
                          kernel_size=3 if dim_change == 'conv3' else 1,
                          stride=stride,
                          padding='half',
                          dilation=1,
                          bias=False)
        else:
            return _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, conv_f, None)


class ResNetV2Unit(E.Seq):
    def __init__(self, block_f=PreactBlock, dim_change='proj', inplace_add=False):
        _check_resnet_unit_args(block_f, dim_change)
        super().__init__()
        self.block_f, self.dim_change, self.inplace_add = block_f, dim_change, inplace_add

    def build(self, x):
        block = self.block_f()
        in_width = x.shape[1]
        block_args = default_args(self.block_f)
        out_width = block_args.base_width * block_args.width_factors[-1]
        stride = block_args.stride
        if stride == 1 and in_width == out_width:
            self.add(fork=E.Fork(block=block, shortcut=nn.Identity()))
        else:
            shortcut = _get_resnetv2_shortcut(in_width, out_width, stride, self.dim_change,
                                              block_args.conv_f)
            self.add(preact=block[:'conv0'],
                     fork=E.Fork(block=block['conv0':], shortcut=shortcut))
        self.add(sum=E.Sum(inplace=self.inplace_add))


class ResNetV2Groups(ResNetV1Groups):
    def __init__(self,
                 group_lengths,
                 base_width,
                 block_f=partial(default_args(ResNetV2Unit).block_f,
                                 base_width=Reserved,
                                 stride=Reserved),
                 dim_change=default_args(ResNetV2Unit).dim_change,
                 unit_f=ResNetV2Unit):
        super().__init__(group_lengths, base_width, block_f, dim_change, unit_f)
        if (norm_f := default_args(block_f).norm_f) is not None:
            self.add('post_norm', norm_f())
        self.add('post_act', default_args(block_f).act_f())


class ResNetV2Backbone(ResNetV1Backbone):
    """Pre-activation resnet backbone."""

    __init__ = partialmethod(ResNetV1Backbone.__init__,
                             block_f=default_args(ResNetV2Groups).block_f,
                             dim_change=default_args(ResNetV2Groups).dim_change,
                             groups_f=ResNetV2Groups)


# DenseNet #########################################################################################


class DenseTransition(E.Seq):
    def __init__(self,
                 compression=0.5,
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=1),
                 noise_f=None,
                 pool_f=partial(E.AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.add(norm=norm_f(), act=act_f())
        if noise_f is not None:
            self.add(noise=noise_f())
        self.args = NameDict(compression=compression, conv_f=conv_f, pool_f=pool_f)

    def build(self, x):
        self.add(conv=self.args.conv_f(out_channels=round(x.shape[1] * self.args.compression)),
                 pool=Reserved.call(self.args.pool_f, stride=2))


class DenseUnit(E.Seq):
    def __init__(self,
                 block_f=tree_partial(PreactBlock,
                                      kernel_sizes=(1, 3),
                                      width_factors=(4, 1),
                                      act_f=ArgTree(inplace=True))):
        super().__init__(fork=E.Fork(skip=E.Identity(), block=block_f()), cat=E.Concat())


class DenseBlock(E.Seq):
    def __init__(self, length, block_f=default_args(DenseUnit).block_f):
        super().__init__({f'unit{i}': DenseUnit(block_f) for i in range(length)})


class DenseSequence(E.Seq):
    def __init__(self,
                 growth_rate,
                 db_lengths,
                 compression=default_args(DenseTransition).compression,
                 block_f=partial(default_args(DenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, length in enumerate(db_lengths):
            self.add(f'db{i}',
                     DenseBlock(length, block_f=Reserved.partial(block_f, base_width=growth_rate)))
            if i != len(db_lengths) - 1:
                self.add(f'transition{i}', DenseTransition(compression, **norm_act_args))
        self.add(norm=default_args(block_f).norm_f(),
                 act=default_args(block_f).act_f())


class DenseNetBackbone(E.Seq):
    def __init__(self,
                 growth_rate=12,
                 small_input=False,
                 db_lengths=(2,) * 4,
                 compression=default_args(DenseSequence).compression,
                 block_f=default_args(DenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=DenseSequence(growth_rate, db_lengths, compression, block_f))


# MDenseNet ########################################################################################


class MDenseTransition(E.Seq):
    def __init__(self,
                 compression=0.5,
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=1),
                 noise_f=None,
                 pool_f=params(DenseTransition).pool_f):
        super().__init__()
        self.args = self.get_args(locals())

    def build(self, x):
        a = self.args
        out_channels = round(sum(y.shape[1] for y in x) * a.compression)
        starts = E.Parallel([E.Seq() for _ in range(len(x))])
        for s in starts:
            s.add(norm=a.norm_f(), act=a.act_f())
            if a.noise_f is not None:
                s.add(noise=a.noise_f())
            s.add(conv=a.conv_f(out_channels=out_channels))
        self.add(starts=starts, sum=E.Sum(), pool=Reserved.call(a.pool_f, stride=2))


class MDenseUnit(E.Module):
    def __init__(self, block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
        super().__init__()
        self.block_f = block_f
        self.block_starts = E.Parallel()
        self.sum = E.Sum()
        block = block_f()
        self._split_index = block.index('conv0') + 1
        self.block_end = block[self._split_index:]

    def build(self, x):
        self.block_starts.extend([self.block_f()[:self._split_index] for _ in range(len(x))])

    def forward(self, x):
        return x + [self.block_end(self.sum(self.block_starts(x)))]


class MDenseBlock(E.Seq):
    def __init__(self, length, block_f=default_args(MDenseUnit).block_f):
        super().__init__(to_list=E.Func(lambda x: [x]),
                         **{f'unit{i}': MDenseUnit(block_f)
                            for i in range(length)})


def mf_dense_sequence_init(module, db_f, transition_f, growth_rate, db_lengths, compression,
                           block_f):
    norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
    for i, len_ in enumerate(db_lengths):
        if i > 0:
            module.add(f'transition{i - 1}', transition_f(compression, **norm_act_args))
        module.add(f'db{i}', db_f(len_, block_f=Reserved.partial(block_f, base_width=growth_rate)))
    module.add('concat', E.Concat())
    module.add(norm=default_args(block_f).norm_f(), act=default_args(block_f).act_f())


class MDenseSequence(E.Seq):
    def __init__(self,
                 growth_rate,
                 db_lengths,
                 compression=default_args(MDenseTransition).compression,
                 block_f=partial(default_args(MDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        mf_dense_sequence_init(self, MDenseBlock, MDenseTransition, growth_rate, db_lengths,
                               compression, block_f)


class MDenseNetBackbone(E.Seq):
    def __init__(self,
                 growth_rate=12,
                 small_input=False,
                 db_lengths=(2,) * 4,
                 compression=default_args(MDenseSequence).compression,
                 block_f=default_args(MDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=MDenseSequence(growth_rate, db_lengths, compression, block_f))


# FDenseNet ########################################################################################

FDenseTransition = MDenseTransition


class FDenseBlock(E.Module):
    def __init__(self,
                 length,
                 block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
        super().__init__()
        self.length = length
        self.width = default_args(block_f).width_factors[0] * default_args(block_f).base_width
        self.sum = E.Sum()
        self.block_start_columns = nn.ModuleList()
        self.block_ends = nn.ModuleList()

        wf = default_args(block_f).width_factors

        def get_width_factors(idx):
            return [wf[0] * (length - idx)] + list(wf[1:])

        split_index = block_f().index('conv0') + 1
        for i in range(length):
            block = block_f(width_factors=get_width_factors(i))
            self.block_start_columns.append(block[:split_index])
            self.block_ends.append(block[self.split_index:])

    def forward(self, x):
        inputs = [x]
        columns = []
        for i, (col, end) in enumerate(zip(self.block_start_columns, self.block_ends)):
            columns.append(col(inputs[-1]))
            inputs.append(
                end(
                    self.sum([
                        columns[j].narrow(1, self.width * (i - j), self.width) for j in range(i + 1)
                    ])))
        return inputs


class FDenseSequence(E.Seq):
    def __init__(self,
                 growth_rate,
                 db_lengths,
                 compression=default_args(FDenseTransition).compression,
                 block_f=partial(default_args(FDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        mf_dense_sequence_init(self, FDenseBlock, FDenseTransition, growth_rate, db_lengths,
                               compression, block_f)


class FDenseNetBackbone(E.Seq):
    def __init__(self,
                 growth_rate=12,
                 small_input=False,
                 db_lengths=(2,) * 4,
                 compression=default_args(FDenseSequence).compression,
                 block_f=default_args(FDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=FDenseSequence(growth_rate, db_lengths, compression, block_f))


# VGG ##############################################################################################


class VGGBackbone(E.Seq):
    def __init__(self,
                 base_width=64,
                 block_depths=(2, 2, 3, 3, 3),
                 block_f=partial(PostactBlock,
                                 kernel_sizes=Reserved,
                                 base_width=Reserved,
                                 width_factors=Reserved,
                                 norm_f=None),
                 pool_f=partial(E.MaxPool, ceil_mode=True)):
        super().__init__()
        for i, d in enumerate(block_depths):
            self.add(
                (f'block{i}',
                 Reserved.call(
                     block_f, kernel_sizes=[3] * d, base_width=base_width,
                     width_factors=[2 ** i] * d)), (f'pool{i}', pool_f(kernel_size=2, stride=2)))


class VGGClassifier(E.Seq):
    def __init__(self, fc_dim, class_count, act_f=E.ReLU, noise_f=nn.Dropout):
        super().__init__()
        widths = [fc_dim] * 2 + [class_count]
        for i, w in enumerate(widths):
            self.add((f'linear{i}', E.Linear(w)), (f'act_fc{i}', act_f()))
            if noise_f:
                self.add(f'noise_fc{i}', noise_f())
        self.add('probs', nn.Softmax(dim=1))


# FCN ##############################################################################################


class FCNEncoder(E.Seq):
    def __init__(self,
                 block_f=default_args(VGGBackbone).block_f,
                 pool_f=partial(E.MaxPool, ceil_mode=True),
                 fc_dim=4096,
                 noise_f=nn.Dropout2d):
        # TODO: do something with pool_f
        super().__init__()
        self.add('vgg_backbone', VGGBackbone(block_f=block_f))
        conv_f = default_args(block_f).conv_f
        act_f = default_args(block_f).act_f
        for i in range(2):
            self.add((f'conv_fc{i}', conv_f(fc_dim, kernel_size=7)), (f'act_fc{i}', act_f()),
                     (f'noise_fc{i}', noise_f()))


# Adversarial autoencoder ##########################################################################
# TODO: linear to batchnorm, output shape


class AAEEncoder(E.Seq):
    def __init__(self,
                 kernel_sizes=(4,) * 3,
                 widths=(64,) + (256,) * 2,
                 z_dim=128,
                 norm_f=D.norm_f,
                 act_f=nn.LeakyReLU,
                 conv_f=D.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add(f'conv{i}', conv_f(out_channels=w, kernel_size=k, stride=2, bias=i == 0))
            if i > 0:
                self.add(f'norm{i}', norm_f())
            self.add(f'act{i}', act_f())
        self.add('linear_z', E.Linear(z_dim))


class AAEDecoder(E.Seq):
    def __init__(self,
                 h_dim=1024,
                 kernel_sizes=(4,) * 3,
                 widths=(256, 128, 1),
                 norm_f=D.norm_f,
                 act_f=E.ReLU,
                 convt_f=D.convt_f):
        super().__init__()
        self.add('linear_h', E.Linear(h_dim))
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add({
                f'norm{i}': norm_f(),
                f'act{i}': act_f(),
                f'conv{i}': convt_f(out_channels=w, kernel_size=k, stride=2, padding=1)
            })
        self.add('tanh', nn.Tanh())


class AAEDiscriminator(E.Seq):
    def __init__(self, h_dim=default_args(AAEDecoder).h_dim, act_f=E.ReLU):
        super().__init__()
        for i in range(2):
            # batch normalization?
            self.add(f'linear{i}', E.Linear(h_dim))
            self.add(f'act{i}', act_f())
        self.add('logits', E.Linear(2))
        self.add('probs', nn.Softmax(dim=1))


# iRevNet ##########################################################################################


class CouplingBaseR(E.Module):
    def __init__(self, first=False, block_f=PreactBlock):
        super().__init__()
        self.args = self.get_args(locals())
        if (w := params(block_f).width_factors[-1]) != 1:
            raise RuntimeError(f"`params(block_f).width_factors[-1]` should be 1, not {w}")
        self.baguette = self.block = None


@E.zero_log_abs_det_jac
class AdditiveCouplingR(CouplingBaseR):
    def build(self, x):
        a, block_args = self.args, params(self.args.block_f)
        self.baguette = Baguette(block_args.stride) if block_args.stride != 1 else E.Identity()
        self.block = a.block_f()['conv0':] if a.first else a.block_f()

    def forward(self, x):
        t = self.block(Ladj.stop(x[1]))
        return self.baguette(x[1]), self.baguette(x[0]) + t

    def inverse_forward(self, y):
        x1 = self.baguette.inverse(y[0])
        t = self.block(Ladj.stop(x1))
        return self.baguette.inverse(y[1] - t), x1


class AffineCouplingD(CouplingBaseR):
    def build(self, x):
        a, block_args = self.args, params(self.args.block_f)
        self.baguette = Baguette(block_args.stride) if block_args.stride != 1 else E.Identity()
        width_factors = list(block_args.width_factors)
        width_factors[-1] *= 2
        block_f = partial(a.block_f, width_factors=width_factors)
        self.block = block_f()['conv0':] if a.first else block_f()

    def forward(self, x):
        bx1 = self.block(Ladj.stop(x[1]))
        t, pre_s = torch.chunk(bx1, 2, dim=1)
        ln_s = pre_s.tanh()
        y = self.baguette(x[1]), self.baguette(x[0]) * ln_s.exp() + t
        return Ladj.add(y, x, lambda: ln_s.view(ln_s.shape[0], -1).sum(1))

    def inverse_forward(self, y):
        if Ladj.has(y[0]):
            raise NotImplementedError()
        x1 = self.baguette.inverse(y[0])
        bx1 = self.block(x1)
        t, pre_s = torch.split(bx1, bx1.shape[1] // 2, dim=1)
        inv_s = pre_s.tanh().neg().exp()  # neg_ sometimes results in an error
        return self.baguette.inverse((y[1] - t) * inv_s), x1


class IRevNetUnit(E.Seq):
    def __init__(self,
                 first=False,
                 block_f=PreactBlock,
                 force_surjection=False,
                 coupling_f=AdditiveCouplingR):
        super().__init__()
        self.args = self.get_args(locals())

    def build(self, x):
        a, ba = self.args, params(self.args.block_f)
        block_out_ch = round(ba.base_width * ba.width_factors[-1])
        padding = self._input_padding(x, block_out_ch, ba.stride, a.force_surjection)
        if padding > 0:
            self.add(input_pad=E.Seq(
                concat=E.Concat(1),  # TODO: make more efficient
                inj_pad=PadChannels(padding),
                psplit=ProportionSplit(0.5, dim=1, rounding=math.floor)))
        # assert padding == 0, (padding, block_out_ch)
        self.add(transform=a.coupling_f(first=a.first, block_f=a.block_f))  # rename to coupling

    @staticmethod
    def _input_padding(x, block_out_ch, stride, force_surjection):
        in_ch = x[0].shape[1] + x[1].shape[1]
        block_in_ch = in_ch * (stride ** 2)
        input_padding = 2 * block_out_ch - block_in_ch
        if input_padding < 0:
            raise RuntimeError(f"The number of output channels of the inner block ({block_out_ch})"
                               f" should be at least {(block_in_ch + 1) // 2}.")
        if input_padding > 0:  # injective i-RevNet
            if stride != 1:
                raise RuntimeError(f"Stride is {stride}, but must be 1 for injective padding.")
            if force_surjection:
                raise RuntimeError(f"Input channel count ({in_ch}) times stride^2 ({stride ** 2})"
                                   f" must equal the output channel count ({block_out_ch}).")
        return input_padding


class IRevNetGroups(E.Seq):
    def __init__(self,
                 group_lengths,
                 base_width,
                 first_stride=1,
                 block_f=partial(default_args(IRevNetUnit).block_f,
                                 base_width=Reserved,
                                 width_factors=(1,) * 2,
                                 stride=Reserved),
                 unit_f=IRevNetUnit):
        super().__init__()
        for i, l in enumerate(group_lengths):
            bw_i = base_width if i == 0 else base_width * 4 ** i
            for j in range(l):
                stride = 1 if j > 0 else first_stride if i == 0 else 2
                u = unit_f(first=(i, j) == (0, 0),
                           block_f=Reserved.partial(block_f, base_width=bw_i, stride=stride))
                self.add(f'unit{i}_{j}', u)


class IRevNetBackbone(E.Seq):
    def __init__(self,
                 init_stride=2,
                 group_lengths=(2,) * 4,
                 base_width=None,
                 block_f=default_args(IRevNetGroups).block_f,
                 groups_f=IRevNetGroups,
                 no_final_postact=False):
        _check_block_args(block_f)
        super().__init__()
        self.args = self.get_args(locals())

    def build(self, x):
        a = self.args
        base_width = a.base_width  # half
        if base_width is None:
            base_width, rem = divmod(x.shape[1] * a.init_stride ** 2, 2)
            if rem != 0:
                raise RuntimeError(f"The number of channels after the first baguette with stride"
                                   f" {a.init_stride} is not even and cannot be split.")
        self.add(baguette=Baguette(a.init_stride),
                 psplit=ProportionSplit(0.5, dim=1, rounding=math.floor),
                 bulk=a.groups_f(a.group_lengths, base_width=base_width, block_f=a.block_f),
                 concat=E.Concat(dim=1))
        if not self.args.no_final_postact:
            if (norm_f := default_args(a.block_f).norm_f) is not None:
                self.add('post_norm', norm_f())
            self.add('post_act', default_args(a.block_f).act_f())
