from functools import partial, partialmethod
from collections.abc import Sequence
import typing as T

import torch
from torch import nn
from torch.nn import functional as F

import vidlu.modules.elements as E
from vidlu.utils.func import params, Reserved, default_args, Empty, ArgTree, argtree_partial
from vidlu.utils.collections import NameDict

from . import _default_factories as D


# Constant functions

class GaussianFilter2D(E.Module):
    def __init__(self, sigma=2, ksize=None, padding_mode='reflect'):
        # TODO: exploit separability of the kernel if it is large
        if ksize is None:
            ksize = 3 * sigma
            ksize += int(ksize % 2 == 0)
        elif ksize % 2 == 0:
            ValueError("`ksize` is required to be odd.")
        super().__init__()
        self.padding = [ksize // 2] * 4
        self.padding_mode = padding_mode

        g = torch.arange(ksize, dtype=torch.float).expand(ksize, ksize)
        g = torch.stack([g, g.t()], dim=-1)
        center = (ksize - 1) / 2
        var = sigma ** 2
        kernel = torch.exp(-g.sub_(center).pow_(2).sum(dim=-1).div_(2 * var))
        self.kernel = kernel.div_(torch.sum(kernel))  # normalize
        self.kernel.requires_grad_(False)

    def forward(self, x):
        kernel = self.kernel.expand(x.shape[1], 1, *self.kernel.shape)
        conv = partial(F.conv2d, weight=kernel, groups=x.shape[1])
        return (conv(F.pad(x, self.padding, mode=self.padding_mode)) if self.padding_mode != 'zeros'
                else conv(x, padding=self.padding))


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


# ResNet/DenseNet root block #######################################################################

class StandardRootBlock(E.Seq):
    """Standard ResNet/DenseNet root block.

    Args:
        out_channels (int): number of output channels.
        small_input (bool): If True, the root block doesn't reduce spatial dimensions. E.g. it
            should be `True` for CIFAR-10, but `False` for ImageNet.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
    """

    def __init__(self, out_channels: int, small_input, norm_f=D.norm_f, act_f=E.ReLU):
        if small_input:  # CIFAR
            super().__init__(conv=E.Conv(out_channels, 3, padding='half', bias=False))
        else:
            super().__init__(conv=E.Conv(out_channels, 7, stride=2, padding='half', bias=False),
                             norm=norm_f(),
                             act=act_f(),
                             pool=E.MaxPool(3, stride=2, padding='half'))


# blocks ###########################################################################################

def _add_norm_act(seq, suffix, norm_f, act_f):
    if norm_f:
        seq.add_module(f'norm{suffix}', norm_f())
    seq.add_module(f'act{suffix}', act_f())


def _resolve_block_args(base_width, width_factors, stride, dilation):
    widths = [base_width * wf for wf in width_factors]
    round_widths = list(map(round, widths))  # fractions to ints
    if widths != round_widths:
        raise ValueError(f"Some widths are not integers (width_factors={width_factors},"
                         f" base_width={base_width}; widths={widths}).")
    widths = round_widths
    conv_defaults = default_args(D.conv_f)
    if not isinstance(stride, Sequence):
        stride = [stride] + [conv_defaults.stride] * (len(widths) - 1)
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

    def __init__(self, *, kernel_sizes, base_width, width_factors, stride=1,
                 dilation=1, noise_locations=(), norm_f=D.norm_f, act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved, dilation=Reserved),
                 noise_f=None):
        super().__init__()
        widths, stride, dilation = _resolve_block_args(base_width, width_factors, stride, dilation)
        for i, (k, w, s, d) in enumerate(zip(kernel_sizes, widths, stride, dilation)):
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=s, dilation=d))


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

    def __init__(self, *, kernel_sizes, base_width, width_factors, stride=1,
                 dilation=1, noise_locations=(), norm_f=D.norm_f, act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved, dilation=Reserved),
                 noise_f=None):
        super().__init__()
        widths, stride, dilation = _resolve_block_args(base_width, width_factors, stride, dilation)
        for i, (k, w, s, d) in enumerate(zip(kernel_sizes, widths, stride, dilation)):
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=s, dilation=d))
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())


# Injective ########################################################################################


class AffineCoupling(E.Module):
    def __init__(self, scale_f, translate_f):
        super().__init__()
        self.scale, self.translate = scale_f(), translate_f()

    def forward(self, x1, x2):
        return x1, x1 * self.scale(x1).exp() + self.translate(x1)

    def inverse_forward(self, y1, y2):
        return y1, (y2 - self.translate(y1)) * (-self.scale(y1)).exp()


class ProportionSplit(E.Module):
    def __init__(self, proportion: float, dim=1):
        super().__init__()

    def forward(self, x):
        return x.split(round(self.args.proportion * x.shape[1]), dim=self.args.dim)

    def make_inverse(self):
        return E.Concat(dim=self.args.dim)


class PadChannels(E.Module):
    def __init__(self, padding):
        padding = [0, padding] if isinstance(padding, int) else list(padding)
        super().__init__()
        if len(padding) != 2:
            raise ValueError("`padding` should be a single `int` or a sequence of length 2.")

    def forward(self, x):  # injective
        return F.pad(x, [0, 0, 0, 0] + self.args.padding)

    def make_inverse(self):
        return UnpadChannels(self.args.padding)


class UnpadChannels(E.Module):
    __init__ = PadChannels.__init__

    def forward(self, x):
        return x[:, self.padding[0]:x.shape[1] - self.padding[1], :, :]

    def make_inverse(self):
        return PadChannels(self.args.padding)


class Baguette(E.Seq):
    """Rearranges a NCHW array into a N(C*b*b)(H/b)(W/b) array.

    From
    https://github.com/jhjacobsen/pytorch-i-revnet/issues/18#issue-486527382"""

    def __init__(self, block_size):
        self.block_size = b = block_size
        if b == 1:
            super().__init__()
        else:
            super().__init__(
                contiginv=E.ContiguousInv(),
                resh1=E.BatchReshape(lambda c, h, w: (c, h // b, b, w // b, b)),
                # n c h/b bh w/b bw
                perm1=E.Permute(0, 3, 5, 1, 2, 4),  # n bh bw c h/b w/b
                resh2=E.BatchReshape(lambda bh, bw, c, h_b, w_b: (bh * bw * c, h_b, w_b)),
                contig=E.Contiguous())

    def __repr__(self):
        return f"{type(self).__name__}({self.block_size})"

    def __str__(self):
        return repr(self)


class SqueezeExcitation(E.Module):
    def __init__(self, channel, reduction=16, squeeze_f=nn.AdaptiveAvgPool2d, act_f=E.ReLU):
        super().__init__()

    def build(self, x):
        a = self.args
        self.attention = E.Seq(
            squeeze=a.squeeze_f(output_size=1),
            lin0=E.Linear(x.shape[1] // self.args.reduction, bias=False),
            act0=self.args.act_f(),
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

    def forward(self, x):
        a = self.args
        return F.interpolate(x, size=a.size, scale_factor=a.scale_factor, mode=a.mode,
                             align_corners=a.align_corners)


# Pyramid pooling ##################################################################################


class DenseSPP(E.Module):
    # Spatial pyramid pooling for dense prediction
    def __init__(self, bottleneck_size=512, level_size=128, out_size=128, grid_sizes=(8, 4, 2, 1),
                 block_f=partial(PreactBlock, base_width=Reserved, kernel_sizes=[1],
                                 width_factors=[1]),
                 upsample=partial(F.interpolate, mode='bilinear', align_corners=False),
                 square_grid=False):
        super().__init__()
        self.grid_sizes = grid_sizes
        self.upsample = upsample
        self.input_block = block_f(base_width=bottleneck_size)  # reduces the number of channels
        self.pyramid = E.ModuleTable(
            {f'block{i}': block_f(base_width=level_size) for i in range(len(grid_sizes))})
        self.fuse_block = block_f(base_width=out_size)
        self.square_grid = square_grid

    def forward(self, x):
        target_size = x.size()[2:4]
        ar = target_size[1] / target_size[0]

        x = self.input_block(x)
        levels = [x]
        for pyr_block, grid_size in zip(self.pyramid, self.grid_sizes):
            if not self.square_grid:  # keep aspect ratio
                grid_size = (grid_size, max(1, round(ar * grid_size)))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)  # TODO: x vs levels[-1]
            level = pyr_block(x_pooled)
            levels.append(self.upsample(level, target_size))
        return self.fuse_block(torch.cat(levels, 1))


# Ladder Upsampling ################################################################################


class LadderUpsampleBlend(E.Seq):
    """ For LadderDenseNet and SwiftNet. """
    _pre_blendings = dict(
        sum=E.Sum,
        concat=E.Concat
    )

    def __init__(self, out_channels, pre_blending='concat',
                 blend_block_f=partial(PreactBlock, base_width=Reserved, width_factors=Reserved)):
        super().__init__()
        if pre_blending not in self._pre_blendings:
            raise ValueError(f"Invalid pre-blending '{pre_blending}'. "
                             + f"It should be one of {tuple(self._pre_blendings.keys())}.")
        block_f = Reserved.partial(blend_block_f, width_factors=[1])
        self.args.block_f = block_f

        self.project = None
        self.pre_blend = self._pre_blendings[pre_blending]()
        self.blend = block_f(base_width=out_channels, kernel_sizes=[3])

    def build(self, x, skip):
        self.project = Reserved.partial(self.args.block_f, base_width=x.shape[1])(kernel_sizes=[1])

    def forward(self, x, skip):
        # resize is not defined in build because it depends on skip.shape
        x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip_proj = self.project(skip)
        b = self.pre_blend(x_up, skip_proj)
        return self.blend(b)


class KresoLadder(E.Module):
    def __init__(self, width, up_blend_f=LadderUpsampleBlend):
        super().__init__()
        self.up_blends = None

    def build(self, x, skips):
        self.up_blends = nn.ModuleList(
            [self.args.up_blend_f(self.args.width) for _ in range(len(skips))])

    def forward(self, x, skips):
        ups = [x]
        for upbl, skip in zip(self.up_blends, skips):
            ups.append(upbl(ups[-1], skip))  # TODO: remove detach
        return ups[-1]


class KresoContext(E.Seq):
    def __init__(self, base_width=128,
                 block_f=partial(PreactBlock, base_width=Reserved, kernel_sizes=Reserved,
                                 width_factors=Reserved, dilation=Reserved)):
        super().__init__(
            context=Reserved.call(block_f, base_width=base_width, kernel_sizes=[1, 3],
                                  width_factors=[4, 1], dilation=[1, 2]))


class KresoLadderNet(E.Module):
    def __init__(self, backbone_f, laterals: T.Sequence[str], ladder_width: int,
                 context_f=DenseSPP, up_blend_f=LadderUpsampleBlend, post_activation=False):
        super().__init__()
        self.backbone = backbone_f()
        self.laterals = laterals
        self.context = context_f()
        self.ladder = KresoLadder(ladder_width, up_blend_f)
        defaults = default_args(default_args(up_blend_f).blend_block_f)
        if post_activation:
            self.norm, self.act = defaults.norm_f(), defaults.act_f()

    def forward(self, x):
        context_input, laterals = E.with_intermediate_outputs(self.backbone, self.laterals)(x)
        context = self.context(context_input)
        ladder_output = self.ladder(context, laterals[::-1])
        return self.act(self.norm(ladder_output)) if self.args.post_activation else ladder_output


# ResNetV1 #########################################################################################

def _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, norm_f):
    if stride == 1 and in_width == out_width:
        return nn.Identity()
    else:
        if dim_change in ('proj', 'conv3'):
            k = 3 if dim_change == 'conv3' else 1
            return E.Seq(
                conv=E.Conv(out_width, kernel_size=k, stride=stride, padding='half', bias=False),
                norm=norm_f())
        else:
            pad = [0] * 5 + [out_width - in_width]
            return E.Seq(pool=E.AvgPool(stride, stride),
                         pad=E.Func(partial(F.pad, pad=pad, mode='constant')))


class ResNetV1Unit(E.Seq):
    def __init__(self, block_f=PostactBlock,
                 dim_change='proj'):
        _check_block_args(block_f)
        super().__init__()
        if dim_change not in ['pad', 'proj', 'conv3']:
            raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")

    def build(self, x):
        block_args = default_args(self.args.block_f)
        shortcut = _get_resnetv1_shortcut(
            in_width=x.shape[1], out_width=block_args.base_width * block_args.width_factors[-1],
            stride=block_args.stride, dim_change=self.args.dim_change, norm_f=block_args.norm_f)
        block_f = self.args.block_f
        block = Reserved.call(block_f)[:-1]  # block without the last activation
        self.add_modules(
            fork=E.Fork(
                shortcut=shortcut,
                block=block),
            sum=E.Sum(),
            act=default_args(block_f).act_f())


class ResNetV1Groups(E.Seq):
    def __init__(self, group_lengths, base_width, width_factors,
                 block_f=partial(default_args(ResNetV1Unit).block_f, base_width=Reserved,
                                 width_factors=Reserved, stride=Reserved),
                 dim_change=default_args(ResNetV1Unit).dim_change, unit_f=ResNetV1Unit):
        super().__init__()
        _check_block_args(block_f)
        for i, l in enumerate(group_lengths):
            for j in range(l):
                u = unit_f(block_f=Reserved.partial(block_f, base_width=base_width * 2 ** i,
                                                    width_factors=width_factors,
                                                    stride=1 + int(i > 0 and j == 0)),
                           dim_change=dim_change)
                self.add_module(f'unit{i}_{j}', u)


class ResNetV1Backbone(E.Seq):
    """ Resnet (V1) backbone.

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

    def __init__(self, base_width=64, small_input=True, group_lengths=(2,) * 4,
                 width_factors=(1,) * 2, block_f=default_args(ResNetV1Groups).block_f,
                 dim_change=default_args(ResNetV1Groups).dim_change, groups_f=ResNetV1Groups):
        _check_block_args(block_f)
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(
            root=StandardRootBlock(base_width, small_input, **norm_act_args),
            bulk=groups_f(group_lengths, base_width=base_width,
                          width_factors=width_factors,
                          block_f=block_f, dim_change=dim_change))


# ResNetV2 #########################################################################################

def _check_block_args(block_f):
    args = params(block_f)
    if 'kernel_sizes' in args and args['kernel_sizes'] is Empty:
        raise ValueError("Argument kernel_sizes missing in block_f.")


def _get_resnetv2_shortcut(in_width, out_width, stride, dim_change):
    if stride == 1 and in_width == out_width:
        return nn.Identity()
    else:
        if dim_change in ('proj', 'conv3'):
            k = 3 if dim_change == 'conv3' else 1
            return E.Conv(out_width, kernel_size=k, stride=stride, padding='half', bias=False)
        else:
            return _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, None)


class ResNetV2Unit(E.Seq):
    def __init__(self, block_f=PreactBlock, dim_change='proj'):
        _check_block_args(block_f)
        super().__init__()
        if dim_change not in ['pad', 'proj', 'conv3']:
            raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")

    def build(self, x):
        block_f = self.args.block_f
        block = block_f()

        dim_change = self.args.dim_change
        in_width = x.shape[1]
        block_args = default_args(self.args.block_f)
        out_width = block_args.base_width * block_args.width_factors[-1]
        stride = block_args.stride
        if stride == 1 and in_width == out_width:
            self.add_module(fork=E.Fork(shortcut=nn.Identity(), block=block))
        else:
            shortcut = _get_resnetv2_shortcut(in_width, out_width, stride, dim_change)
            self.add_modules(preact=block[:'conv0'],
                             fork=E.Fork(shortcut=shortcut, block=block['conv0':]))
        self.add_module(sum=E.Sum())


class ResNetV2Groups(ResNetV1Groups):
    def __init__(self, group_lengths, base_width, width_factors,
                 block_f=partial(default_args(ResNetV2Unit).block_f, base_width=Reserved,
                                 width_factors=Reserved, stride=Reserved),
                 dim_change=default_args(ResNetV2Unit).dim_change):
        super().__init__(group_lengths, base_width, width_factors, block_f, dim_change,
                         unit_f=ResNetV2Unit)
        norm_f = default_args(block_f).norm_f
        if norm_f is not None:
            self.add_module('post_norm', norm_f())
        self.add_module('post_act', default_args(block_f).act_f())


class ResNetV2Backbone(ResNetV1Backbone):
    """ Pre-activation resnet backbone.

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

    __init__ = partialmethod(ResNetV1Backbone.__init__,
                             block_f=default_args(ResNetV2Groups).block_f,
                             dim_change=default_args(ResNetV2Groups).dim_change,
                             groups_f=ResNetV2Groups)


# DenseNet #########################################################################################

class DenseTransition(E.Seq):
    def __init__(self, compression=0.5, norm_f=D.norm_f, act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(E.AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.add_modules(norm=norm_f(),
                         act=act_f())
        if self.args.noise_f is not None:
            self.add_module(noise=noise_f())
        self.args = NameDict(compression=compression, conv_f=conv_f, pool_f=pool_f)

    def build(self, x):
        self.add_modules(
            conv=self.args.conv_f(out_channels=round(x.shape[1] * self.args.compression)),
            pool=Reserved.call(self.args.pool_f, stride=2))


class DenseUnit(E.Seq):
    def __init__(self,
                 block_f=argtree_partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1),
                                         act_f=ArgTree(inplace=True))):
        super().__init__(fork=E.Fork(skip=E.Identity(), block=block_f()), cat=E.Concat())


class DenseBlock(E.Seq):
    def __init__(self, length, block_f=default_args(DenseUnit).block_f):
        super().__init__({f'unit{i}': DenseUnit(block_f) for i in range(length)})


class DenseSequence(E.Seq):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(DenseTransition).compression,
                 block_f=partial(default_args(DenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, length in enumerate(db_lengths):
            self.add_module(f'db{i}',
                            DenseBlock(length,
                                       block_f=Reserved.partial(block_f, base_width=growth_rate)))
            if i != len(db_lengths) - 1:
                self.add_module(f'transition{i}', DenseTransition(compression, **norm_act_args))
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class DenseNetBackbone(E.Seq):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(DenseSequence).compression,
                 block_f=default_args(DenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=DenseSequence(growth_rate, db_lengths, compression, block_f))


# MDenseNet ########################################################################################

class MDenseTransition(E.Seq):
    def __init__(self, compression=0.5, norm_f=D.norm_f, act_f=E.ReLU,
                 conv_f=partial(D.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(E.AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.args = NameDict(compression=compression, norm_f=norm_f, act_f=act_f, conv_f=conv_f,
                             noise_f=noise_f, pool_f=pool_f)

    def build(self, x):
        a = self.args
        out_channels = round(sum(y.shape[1] for y in x) * a.compression)
        starts = E.Parallel([E.Seq() for _ in range(len(x))])
        for s in starts:
            s.add_modules(norm=a.norm_f(),
                          act=a.act_f())
            if a.noise_f is not None:
                s.add_module(noise=a.noise_f())
            s.add_module(conv=a.conv_f(out_channels=out_channels))
        self.add_modules(starts=starts,
                         sum=E.Sum(),
                         pool=Reserved.call(a.pool_f, stride=2))


class MDenseUnit(E.Module):
    def __init__(self, block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
        super().__init__()
        self.block_starts = E.Parallel()
        self.sum = E.Sum()
        block = block_f()
        self._split_index = block.index('conv0') + 1
        self.block_end = block[self._split_index:]

    def build(self, x):
        self.block_starts.extend([self.args.block_f()[:self._split_index] for _ in range(len(x))])

    def forward(self, x):
        return x + [self.block_end(self.sum(self.block_starts(x)))]


class MDenseBlock(E.Seq):
    def __init__(self, length, block_f=default_args(MDenseUnit).block_f):
        super().__init__(to_list=E.Func(lambda x: [x]),
                         **{f'unit{i}': MDenseUnit(block_f) for i in range(length)})


class MDenseSequence(E.Seq):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(MDenseTransition).compression,
                 block_f=partial(default_args(MDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, len_ in enumerate(db_lengths):
            if i > 0:
                self.add_module(f'transition{i - 1}',
                                MDenseTransition(compression, **norm_act_args))
            self.add_module(f'db{i}',
                            MDenseBlock(len_,
                                        block_f=Reserved.partial(block_f, base_width=growth_rate)))
        self.add_module('concat', E.Concat())
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class MDenseNetBackbone(E.Seq):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(MDenseSequence).compression,
                 block_f=default_args(MDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=MDenseSequence(growth_rate, db_lengths, compression, block_f))


# FDenseNet ########################################################################################

FDenseTransition = MDenseTransition


class FDenseBlock(E.Module):
    def __init__(self, length,
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
            inputs.append(end(self.sum(
                [columns[j].narrow(1, self.width * (i - j), self.width)
                 for j in range(i + 1)])))
        return inputs


class FDenseSequence(E.Seq):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(FDenseTransition).compression,
                 block_f=partial(default_args(FDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, len_ in enumerate(db_lengths):
            if i > 0:
                self.add_module(f'transition{i - 1}',
                                FDenseTransition(compression, **norm_act_args))
            self.add_module(f'db{i}',
                            FDenseBlock(len_,
                                        block_f=Reserved.partial(block_f, base_width=growth_rate)))
        self.add_module('concat', E.Concat())
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class FDenseNetBackbone(E.Seq):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(FDenseSequence).compression,
                 block_f=default_args(FDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=StandardRootBlock(2 * growth_rate, small_input, **norm_act_args),
                         bulk=FDenseSequence(growth_rate, db_lengths, compression, block_f))


# VGG ##############################################################################################

class VGGBackbone(E.Seq):
    def __init__(self, base_width=64, block_depths=(2, 2, 3, 3, 3),
                 block_f=partial(PostactBlock, kernel_sizes=Reserved, base_width=Reserved,
                                 width_factors=Reserved, norm_f=None),
                 pool_f=partial(E.MaxPool, ceil_mode=True)):
        super().__init__()
        for i, d in enumerate(block_depths):
            self.add_modules(
                (f'block{i}', Reserved.call(block_f, kernel_sizes=[3] * d, base_width=base_width,
                                            width_factors=[2 ** i] * d)),
                (f'pool{i}', pool_f(kernel_size=2, stride=2)))


class VGGClassifier(E.Seq):
    def __init__(self, fc_dim, class_count, act_f=E.ReLU, noise_f=nn.Dropout):
        super().__init__()
        widths = [fc_dim] * 2 + [class_count]
        for i, w in enumerate(widths):
            self.add_modules((f'linear{i}', E.Linear(w)),
                             (f'act_fc{i}', act_f()))
            if noise_f:
                self.add_module(f'noise_fc{i}', noise_f())
        self.add_module('probs', nn.Softmax(dim=1))


# FCN ##############################################################################################

class FCNEncoder(E.Seq):
    def __init__(self, block_f=default_args(VGGBackbone).block_f,
                 pool_f=partial(E.MaxPool, ceil_mode=True), fc_dim=4096, noise_f=nn.Dropout2d):
        super().__init__()
        self.add_module('vgg_backbone', VGGBackbone(block_f=block_f))
        conv_f = default_args(block_f).conv_f
        act_f = default_args(block_f).act_f
        for i in range(2):
            self.add_modules((f'conv_fc{i}', conv_f(fc_dim, kernel_size=7)),
                             (f'act_fc{i}', act_f()),
                             (f'noise_fc{i}', noise_f()))


# Autoencoder ######################################################################################

class SimpleEncoder(E.Seq):
    def __init__(self, kernel_sizes=(4,) * 4, widths=(32, 64, 128, 256), z_width=32,
                 norm_f=D.norm_f, act_f=E.ReLU, conv_f=D.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if norm_f is not None:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', E.Linear(z_width))


# Adversarial autoencoder ##########################################################################
# TODO: linear to batchnorm, output shape

class AAEEncoder(E.Seq):
    def __init__(self, kernel_sizes=(4,) * 3, widths=(64,) + (256,) * 2, z_dim=128,
                 norm_f=D.norm_f, act_f=nn.LeakyReLU, conv_f=D.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if i > 0:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', E.Linear(z_dim))


class AAEDecoder(E.Seq):
    def __init__(self, h_dim=1024, kernel_sizes=(4,) * 3, widths=(256, 128, 1),
                 norm_f=D.norm_f, act_f=E.ReLU, convt_f=D.convt_f):
        super().__init__()
        self.add_module('linear_h', E.Linear(h_dim))
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module({f'norm{i}': norm_f(),
                             f'act{i}': act_f(),
                             f'conv{i}': convt_f(kernel_size=k, out_channels=w, stride=2,
                                                 padding=1)})
        self.add_module('tanh', nn.Tanh())


class AAEDiscriminator(E.Seq):
    def __init__(self, h_dim=default_args(AAEDecoder).h_dim, norm_f=D.norm_f, act_f=E.ReLU):
        super().__init__()
        for i in range(2):
            # batch normalization?
            self.add_module(f'linear{i}', E.Linear(h_dim))
            self.add_module(f'act{i}', act_f())
        self.add_module('logits', E.Linear(2))
        self.add_module('probs', nn.Softmax(dim=1))


# iRevNet ##########################################################################################

def _get_irevnet_unit_input_padding(x, block_out_ch, stride, force_bijection):
    half_in_ch = x[0].shape[1]
    if not x[1].shape[1] == half_in_ch:
        raise RuntimeError(f"Shapes of both inputs must match ({x[0].shape} != {x[1].shape})")
    input_padding = 2 * (block_out_ch - half_in_ch * (stride ** 2))
    if input_padding < 0:
        raise RuntimeError(f"The number of output channels of the inner block ({block_out_ch})"
                           f" should be at least {half_in_ch}.")
    if input_padding > 0:  # injective i-RevNet
        if stride != 1:
            raise RuntimeError(f"Stride is {stride}, but must be 1 for injective padding.")
        if force_bijection:
            raise RuntimeError("For surjectiveness, the number of channels in the output of the"
                               + f" inner block must be {half_in_ch}.")
    return input_padding


class IRevNetUnit(E.Module):
    def __init__(self, first=False, block_f=PreactBlock, force_surjection=False):
        super().__init__()
        self.input_pad, self.baguette, self.block = [None] * 3

    def build(self, x):
        a = self.args
        stride = params(a.block_f).stride
        block_out_ch = round(params(a.block_f).base_width * params(a.block_f).width_factors[-1])
        input_padding = _get_irevnet_unit_input_padding(x, block_out_ch, stride, a.force_surjection)
        self.input_pad = (E.Identity() if self.input_pad == 0
                          else E.Seq(concat=E.Concat(1),
                                     inj_pad=PadChannels(input_padding),
                                     split=ProportionSplit(0.5, dim=1)))
        self.baguette = Baguette(stride)
        self.block = a.block_f()['conv0':] if a.first else a.block_f()
        # self.register_forward_hook(lambda module, input, output: print(output[0].shape) )

    def forward(self, x):
        h = self.input_pad(x)
        return self.baguette(h[1]), self.block(h[1]) + self.baguette(h[0])

    def inverse_forward(self, y):
        h1 = self.baguette.inverse(y[0])
        h0 = self.baguette.inverse(y[1] - self.block(h1))
        return self.input_pad.inverse([h0, h1])


class IRevNetGroups(E.Seq):
    def __init__(self, group_lengths, base_width, width_factors, first_stride=1,
                 block_f=partial(default_args(IRevNetUnit).block_f, base_width=Reserved,
                                 width_factors=Reserved, stride=Reserved),
                 unit_f=IRevNetUnit):
        super().__init__()
        for i, l in enumerate(group_lengths):
            base_width_i = base_width * ((1 + int(i > 0)) ** 2) ** i
            for j in range(l):
                stride = (first_stride if i == 0 else 2) if j == 0 else 1
                u = unit_f(block_f=Reserved.partial(block_f, base_width=base_width_i,
                                                    width_factors=width_factors, stride=stride),
                           first=(i, j) == (0, 0))
                self.add_module(f'unit{i}_{j}', u)


class IRevNetBackbone(E.Seq):
    def __init__(self, init_stride=2, group_lengths=(2,) * 4, width_factors=(1,) * 2,
                 base_width=None, block_f=default_args(IRevNetGroups).block_f,
                 groups_f=IRevNetGroups):
        _check_block_args(block_f)
        super().__init__()

    def build(self, x):
        a = self.args
        base_width = a.base_width
        if base_width is None:
            base_width, rem = divmod(x.shape[0] * a.init_stride ** 2, 2)
            if rem != 0:
                raise RuntimeError(f"The number of channels after the first baguette with"
                                   f" stride {a.init_stride} is not even.")
        _check_block_args(a.block_f)
        self.add_module('baguette', Baguette(a.init_stride))
        self.add_module("split", ProportionSplit(0.5, dim=1))
        self.add_module('bulk', a.groups_f(a.group_lengths, base_width=base_width,
                                           width_factors=a.width_factors,
                                           block_f=a.block_f))
        self.add_module('concat', E.Concat(dim=1))
        norm_f = default_args(a.block_f).norm_f
        if norm_f is not None:
            self.add_module('post_norm', norm_f())
        self.add_module('post_act', default_args(a.block_f).act_f())


def split(x):
    n = int(x.size()[1] / 2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = PadChannels(self.pad)
        self.psi = Baguette(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []
        from fractions import Fraction as Frac
        self.input_pad = (E.Identity() if self.pad == 0
                          else E.Seq(concat=E.Concat(1),
                                     inj_pad=PadChannels(self.pad),
                                     split=ProportionSplit(0.5, dim=1)))
        self.bottleneck_block = PreactBlock(base_width=out_ch,
                                            width_factors=[Frac(1, 4), Frac(1, 4), 1],
                                            kernel_sizes=[3, 3, 3], stride=stride)
        if first:
            self.bottleneck_block = self.bottleneck_block['conv0':]

        # self.unit = IRevNetUnit(first=first,
        #                        block_f=partial(PreactBlock, base_width=out_ch,
        #                                        width_factors=[Frac(1, 4), Frac(1, 4), 1],
        #                                        kernel_sizes=[3, 3, 3], stride=stride))

    def forward(self, x):
        # return self.unit(x)
        """ bijective or injective block forward """
        if self.pad != 0 and self.stride == 1:
            # x = self.input_pad.concat(x[0], x[1])
            # x = self.input_pad.inj_pad(x)
            # x = self.input_pad.split(x)
            # for m in self.input_pad._modules.values():
            #    x=m(x)
            x = self.input_pad(x)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNetBackboneOrig(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super().__init__()
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch // 2, self.in_ch // 2 * 4,
                         self.in_ch // 2 * 4 ** 2, self.in_ch // 2 * 4 ** 3]

        self.init_psi = Baguette(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        self.bn1 = nn.BatchNorm2d(nChannels[-1] * 2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1] * 2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch // 2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(out_bij))
        return out
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1 - i].inverse(out)
        out = merge(out[0], out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return x


class iRevNetBackboneHyb(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super().__init__()
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch // 2, self.in_ch // 2 * 4,
                         self.in_ch // 2 * 4 ** 2, self.in_ch // 2 * 4 ** 3]

        self.init_psi = Baguette(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        self.bn1 = nn.BatchNorm2d(nChannels[-1] * 2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1] * 2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch // 2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(out_bij))
        return out
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij
