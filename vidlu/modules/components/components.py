from functools import partial, partialmethod
from collections import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from vidlu.modules.elements import (Module, Sequential, ModuleTable, Conv, MaxPool, Linear, Func,
                                    AvgPool, Sum, Concat, Identity, Branching, Parallel)
import vidlu.modules.elements as E
from vidlu.utils.func import params, Reserved, default_args, Empty, ArgTree, argtree_partial
from vidlu.utils.collections import NameDict

from . import _default_factories as D


# Default arguments ################################################################################


def _call_no_inplace(module_f):
    if 'inplace' in params(module_f):
        return module_f(inplace=False)
    return module_f()


# ResNet/DenseNet root block #######################################################################

class RootBlock(Sequential):
    """Standard ResNet/DenseNet root block.

    Args:
        out_channels (int): number of output channels.
        small_input (bool): If True, the root block doesn't reduce spatial dimensions. E.g. it
            should be `True` for CIFAR-10, but `False` for ImageNet.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
    """

    def __init__(self, out_channels: int, small_input, norm_f=D.norm_f, act_f=D.act_f):
        if small_input:  # CIFAR
            super().__init__(conv=Conv(out_channels, 3, padding='half', bias=False))
        else:
            super().__init__(conv=Conv(out_channels, 7, stride=2, padding='half', bias=False),
                             norm=norm_f(),
                             activation=act_f(),
                             pool=MaxPool(3, stride=2, padding='half'))


# blocks ###########################################################################################

def _add_norm_act(seq, suffix, norm_f, act_f):
    if norm_f:
        seq.add_module(f'norm{suffix}', norm_f())
    seq.add_module(f'act{suffix}', act_f())


class PreactBlock(Sequential):
    """A block of one or more sequences of the form
    "normalization -> activation [-> noise] -> convolution".

    Args:
        kernel_sizes (Sequence[int]): Kernel sizes of convolutions.
        base_width (int): Base width that is multiplied by a width factor from
            `width_factors` to get its output widths for all convolutions.
        width_factors (Sequence[int]): Factors that multiply `base_width` to get
            the output width for all convolutions.
        noise_locations (Sequence[int]): The indices of convolutions that are to
            be followed by a noise layer.
        stride (int): Stride of the first convolution.
        omit_first_preactivation (bool): If True, the first normalization and
            activation are to be omitted.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
        conv_f: Convolution module factory.
        noise_f: Noise module factory.
    """

    def __init__(self, *, kernel_sizes, base_width, width_factors, noise_locations=(), stride=1,
                 dilation=1, norm_f=D.norm_f, act_f=D.act_f,
                 conv_f=partial(D.conv_f, stride=Reserved, dilation=Reserved, kernel_size=Reserved,
                                out_channels=Reserved),
                 noise_f=None):
        super().__init__()
        widths = [base_width * wf for wf in width_factors]
        conv_defaults = default_args(D.conv_f)
        if not isinstance(stride, Sequence):
            stride = [stride] + [conv_defaults.stride] * (len(widths) - 1)
        if not isinstance(dilation, Sequence):
            dilation = [dilation] + [conv_defaults.dilation] * (len(widths) - 1)
        for i, (k, w, s, d) in enumerate(zip(kernel_sizes, widths, stride, dilation)):
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=s, dilation=d))


class PostactBlock(Sequential):
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
        stride (int): Stride of the first convolution.
        conv_f: Convolution module factory.
        norm_f: Normalization module factory.
        act_f: Activation module factory.
        noise_f: Noise module factory.
    """

    def __init__(self, *, kernel_sizes, base_width, width_factors, noise_locations=(), stride=1,
                 norm_f=D.norm_f, act_f=D.act_f,
                 conv_f=partial(D.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved),
                 noise_f=None):
        super().__init__()
        widths = [base_width * wf for wf in width_factors]
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=stride if i == 0 else 1))
            _add_norm_act(self, f'{i}', norm_f, act_f)
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())


# Invertible #######################################################################################

class AffineCoupling(Module):
    def __init__(self, scale_f, translate_f):
        super().__init__()
        self.scale, self.translate = scale_f(), translate_f()

    def forward(self, x1, x2):
        x1, x1 * self.scale(x1).exp() + self.translate(x1)

    def invert(self, y1, y2):
        return y1, (y2 - self.translate(y1)) * (-self.scale(y1)).exp()


# Resizing #########################################################################################

class Resize(Module):
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


class DenseSPP(Module):
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
        self.pyramid_blocks = ModuleTable(
            {f'block{i}': block_f(base_width=level_size) for i in range(len(grid_sizes))})
        self.fuse_block = block_f(base_width=out_size)
        self.square_grid = square_grid

    def forward(self, x):
        target_size = x.size()[2:4]
        ar = target_size[1] / target_size[0]

        x = self.input_block(x)
        levels = [x]
        for pyr_block, grid_size in zip(self.pyramid_blocks, self.grid_sizes):
            if not self.square_grid:  # keep aspect ratio
                grid_size = (grid_size, max(1, round(ar * grid_size)))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)  # TODO: x vs levels[-1]
            level = pyr_block(x_pooled)
            levels.append(self.upsample(level, target_size))
        return self.fuse_block(torch.cat(levels, 1))


# Ladder Upsampling ################################################################################


class LadderUpsampleBlend(Sequential):
    """ For LadderDenseNet and SwiftNet. """
    _pre_blendings = dict(
        sum=Sum,
        concat=Concat
    )

    def __init__(self, out_channels, pre_blending='concat',
                 blend_block_f=partial(PreactBlock, base_width=Reserved, width_factors=Reserved)):
        super().__init__()
        if pre_blending not in self._pre_blendings:
            raise ValueError(f"Invalid pre-blending '{pre_blending}'. "
                             + f"It should be one of {tuple(self._pre_blendings.keys())}.")
        self.args.block_f = Reserved.partial(blend_block_f, width_factors=[1])

    def build(self, x, skip):
        a = self.args
        self.add_modules(
            parallel=Parallel(
                upsample=Resize(size=skip.shape[2:], mode='bilinear'),
                project=Reserved.partial(a.block_f, base_width=x.shape[1])(kernel_sizes=[1])),
            pre_blend=self._pre_blendings[a.pre_blending](),
            blend=a.block_f(base_width=a.out_channels, kernel_sizes=[3]))

    def forward(self, x, skip):
        return super().forward((x, skip))


class KresoLadder(Module):
    def __init__(self, width, upsample_blend_f=LadderUpsampleBlend):
        super().__init__()

    def build(self, inputs):
        self.upsample_blends = nn.ModuleList(
            [self.args.upsample_blend_f(self.args.width) for _ in range(len(inputs) - 1)])

    def forward(self, skips):
        ups = [skips[0]]
        for upbl, skip in zip(self.upsample_blends, skips):
            ups.append(upbl(ups[-1], skip))
        return ups[-1]


class KresoContext(Sequential):
    def __init__(self, base_width=128,
                 block_f=partial(PreactBlock, base_width=Reserved, kernel_sizes=Reserved,
                                 width_factors=Reserved, dilation=Reserved)):
        super().__init__(
            context=Reserved.call(block_f, base_width=base_width, kernel_sizes=[1, 3],
                                  width_factors=[4, 1], dilation=[1, 2]))


class KresoLadderNet(Module):
    def __init__(self, backbone_f, intermediate_paths, ladder_width, context_f=DenseSPP,
                 upsample_blend_f=LadderUpsampleBlend):
        super().__init__()
        self.backbone_intermediate = E.IntermediateOutputsModuleWrapper(backbone_f(),
                                                                        intermediate_paths)
        self.context = context_f()
        self.ladder = KresoLadder(ladder_width, upsample_blend_f)
        defaults = default_args(default_args(upsample_blend_f).blend_block_f)
        self.norm, self.act = defaults.norm_f(), defaults.act_f()

    def forward(self, x):
        backbone_outputs = self.backbone_intermediate(x)
        ladder_inputs = (self.context(backbone_outputs[0]),) + backbone_outputs[1][::-1]
        return self.act(self.norm(self.ladder(ladder_inputs)))


# ResNetV1 #########################################################################################

def _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, norm_f):
    if stride == 1 and in_width == out_width:
        return Identity()
    else:
        if dim_change in ('proj', 'conv3'):
            k = 3 if dim_change == 'conv3' else 1
            return Sequential(
                conv=Conv(out_width, kernel_size=k, stride=stride, padding='half', bias=False),
                norm=norm_f())
        else:
            pad = [0] * 5 + [out_width - in_width]
            return Sequential(pool=AvgPool(stride, stride),
                              pad=Func(partial(F.pad, pad=pad, mode='constant')))


"""
class ResNetV1UnitOld(Module):
    def __init__(self, block_f=partial(PostactBlock, omit_first_preactivation=Reserved),
                 dim_change='proj'):
        _check_block_args(block_f)
        super().__init__()
        if dim_change not in ['pad', 'proj', 'conv3']:
            raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")
        self.block = Reserved.call(block_f, omit_first_preactivation=False)[:-1]
        self.last_act = default_args(block_f).act_f()
        self.shortcut = None

    def build(self, x):
        block_args = default_args(self.args.block_f)
        out_width = block_args.base_width * block_args.width_factors[-1]
        stride = block_args.stride
        if stride == 1 and x.shape[1] == out_width:
            self.shortcut = Identity()
        else:
            if self.args.dim_change in ('proj', 'conv3'):
                k = 3 if self.args.dim_change == 'conv3' else 1
                self.shortcut = Conv(out_width, kernel_size=k, stride=stride, padding='half',
                                     bias=False)
            else:
                pad = [0] * 5 + [out_width - x.shape[1]]
                self.shortcut = Sequential(pool=AvgPool(stride, stride),
                                           pad=Func(partial(F.pad, pad=pad, mode='constant')))

    def forward(self, x):
        return self.last_act(self.block(x) + self.shortcut(x))
"""


class ResNetV1Unit(Sequential):
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
            branchout=Branching(
                shortcut=shortcut,
                block=block),
            sum=Sum(),
            act=default_args(block_f).act_f())


class ResNetV1Groups(Sequential):
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


class ResNetV1Backbone(Sequential):
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
            root=RootBlock(base_width, small_input, **norm_act_args),
            features=groups_f(group_lengths, base_width=base_width,
                              width_factors=width_factors,
                              block_f=block_f, dim_change=dim_change))


# ResNetV2 #########################################################################################

def _check_block_args(block_f):
    args = params(block_f)
    if 'kernel_sizes' in args and args['kernel_sizes'] is Empty:
        raise ValueError("Argument kernel_sizes missing in block_f.")


def _get_resnetv2_shortcut(in_width, out_width, stride, dim_change):
    if stride == 1 and in_width == out_width:
        return Identity()
    else:
        if dim_change in ('proj', 'conv3'):
            k = 3 if dim_change == 'conv3' else 1
            return Conv(out_width, kernel_size=k, stride=stride, padding='half', bias=False)
        else:
            return _get_resnetv1_shortcut(in_width, out_width, stride, dim_change, None)


"""
class ResNetV2UnitOld(Module):
    def __init__(self, block_f=partial(PreactBlock, omit_first_preactivation=Reserved),
                 dim_change='proj'):
        _check_block_args(block_f)
        super().__init__()
        if dim_change not in ['pad', 'proj', 'conv3']:
            raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")
        block = Reserved.call(block_f, omit_first_preactivation=False)
        self.preact = block[:'conv0']
        self.block = block['conv0':]
        self.shortcut = None
        del block

    build = ResNetV1Unit.build

    def forward(self, x):
        p = self.preact(x)
        return self.block(p) + self.shortcut(p if self.args.dim_change in ('conv3', 'proj') else x)
"""


class ResNetV2Unit(Sequential):
    def __init__(self, block_f=PreactBlock,                 dim_change='proj'):
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
            self.add_module(branching=Branching(shortcut=Identity(), block=block))
        else:
            shortcut = _get_resnetv2_shortcut(in_width, out_width, stride, dim_change)
            self.add_modules(preact=block[:'conv0'],
                             branching=Branching(shortcut=shortcut, block=block['conv0':]))
        self.add_module(sum=Sum())


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

class DenseTransition(Sequential):
    def __init__(self, compression=0.5, norm_f=D.norm_f, act_f=D.act_f,
                 conv_f=partial(D.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(AvgPool, kernel_size=2, stride=Reserved)):
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


class DenseUnit(Module):
    def __init__(self,
                 block_f=argtree_partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1),
                                         act_f=ArgTree(inplace=True))):
        super().__init__()
        self.block = block_f()

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class DenseBlock(Sequential):
    def __init__(self, length, block_f=default_args(DenseUnit).block_f):
        super().__init__({f'unit{i}': DenseUnit(block_f) for i in range(length)})


class DenseSequence(Sequential):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(DenseTransition).compression,
                 block_f=partial(default_args(DenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, length in enumerate(db_lengths):
            self.add_module(f'dense_block{i}',
                            DenseBlock(length,
                                       block_f=Reserved.partial(block_f, base_width=growth_rate)))
            if i != len(db_lengths) - 1:
                self.add_module(f'transition{i}', DenseTransition(compression, **norm_act_args))
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class DenseNetBackbone(Sequential):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(DenseSequence).compression,
                 block_f=default_args(DenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=RootBlock(2 * growth_rate, small_input, **norm_act_args),
                         features=DenseSequence(growth_rate, db_lengths, compression, block_f))


# MDenseNet ########################################################################################

class MDenseTransition(Sequential):
    def __init__(self, compression=0.5, norm_f=D.norm_f, act_f=D.act_f,
                 conv_f=partial(D.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.args = NameDict(compression=compression, norm_f=norm_f, act_f=act_f, conv_f=conv_f,
                             noise_f=noise_f, pool_f=pool_f)

    def build(self, x):
        a = self.args
        out_channels = round(sum(y.shape[1] for y in x) * a.compression)
        starts = Parallel([Sequential() for _ in range(len(x))])
        for s in starts:
            s.add_modules(norm=a.norm_f(),
                          act=a.act_f())
            if a.noise_f is not None:
                s.add_module(noise=a.noise_f())
            s.add_module(conv=a.conv_f(out_channels=out_channels))
        self.add_modules(starts=starts,
                         sum=Sum(),
                         pool=Reserved.call(a.pool_f, stride=2))


class MDenseUnit(Module):
    def __init__(self, block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
        super().__init__()
        self.block_starts = Parallel()
        self.sum = Sum()
        block = block_f()
        self._split_index = block.index('conv0') + 1
        self.block_end = block[self._split_index:]

    def build(self, x):
        self.block_starts.extend([self.args.block_f()[:self._split_index] for _ in range(len(x))])

    def forward(self, x):
        return x + [self.block_end(self.sum(self.block_starts(x)))]


class MDenseBlock(Sequential):
    def __init__(self, length, block_f=default_args(MDenseUnit).block_f):
        super().__init__(to_list=Func(lambda x: [x]),
                         **{f'unit{i}': MDenseUnit(block_f) for i in range(length)})


class MDenseSequence(Sequential):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(MDenseTransition).compression,
                 block_f=partial(default_args(MDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, len in enumerate(db_lengths):
            if i > 0:
                self.add_module(f'transition{i - 1}',
                                MDenseTransition(compression, **norm_act_args))
            self.add_module(f'dense_block{i}',
                            MDenseBlock(len,
                                        block_f=Reserved.partial(block_f, base_width=growth_rate)))
        self.add_module('concat', Concat())
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class MDenseNetBackbone(Sequential):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(MDenseSequence).compression,
                 block_f=default_args(MDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=RootBlock(2 * growth_rate, small_input, **norm_act_args),
                         features=MDenseSequence(growth_rate, db_lengths, compression, block_f))


# FDenseNet ########################################################################################

FDenseTransition = MDenseTransition


class FDenseBlock(Module):
    def __init__(self, length,
                 block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
        super().__init__()
        self.length = length
        self.width = default_args(block_f).width_factors[0] * default_args(block_f).base_width
        self.sum = Sum()
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


class FDenseSequence(Sequential):
    def __init__(self, growth_rate, db_lengths,
                 compression=default_args(FDenseTransition).compression,
                 block_f=partial(default_args(FDenseBlock).block_f, base_width=Reserved)):
        super().__init__()
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, len in enumerate(db_lengths):
            if i > 0:
                self.add_module(f'transition{i - 1}',
                                FDenseTransition(compression, **norm_act_args))
            self.add_module(f'dense_block{i}',
                            FDenseBlock(len,
                                        block_f=Reserved.partial(block_f, base_width=growth_rate)))
        self.add_module('concat', Concat())
        self.add_modules(norm=default_args(block_f).norm_f(),
                         act=default_args(block_f).act_f())


class FDenseNetBackbone(Sequential):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(FDenseSequence).compression,
                 block_f=default_args(FDenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=RootBlock(2 * growth_rate, small_input, **norm_act_args),
                         features=FDenseSequence(growth_rate, db_lengths, compression, block_f))


# VGG ##############################################################################################

class VGGBackbone(Sequential):
    def __init__(self, base_width=64, block_depths=(2, 2, 3, 3, 3),
                 block_f=partial(PostactBlock, kernel_sizes=Reserved, base_width=Reserved,
                                 width_factors=Reserved, norm_f=None),
                 pool_f=partial(MaxPool, ceil_mode=True)):
        super().__init__()
        for i, d in enumerate(block_depths):
            self.add_modules(
                (f'block{i}', Reserved.call(block_f, kernel_sizes=[3] * d, base_width=base_width,
                                            width_factors=[2 ** i] * d)),
                (f'pool{i}', pool_f(kernel_size=2, stride=2)))


class VGGClassifier(Sequential):
    def __init__(self, fc_dim, class_count, act_f=D.act_f, noise_f=nn.Dropout):
        super().__init__()
        widths = [fc_dim] * 2 + [class_count]
        for i, w in enumerate(widths):
            self.add_modules((f'linear{i}', Linear(w)),
                             (f'act_fc{i}', act_f()))
            if noise_f:
                self.add_module(f'noise_fc{i}', noise_f())
        self.add_module('probs', nn.Softmax(dim=1))


# FCN ##############################################################################################

class FCNEncoder(Sequential):
    def __init__(self, block_f=default_args(VGGBackbone).block_f,
                 pool_f=partial(MaxPool, ceil_mode=True), fc_dim=4096, noise_f=nn.Dropout2d):
        super().__init__()
        self.add_module('vgg_backbone', VGGBackbone(block_f=block_f))
        conv_f = default_args(block_f).conv_f
        act_f = default_args(block_f).act_f
        for i in range(2):
            self.add_modules((f'conv_fc{i}', conv_f(fc_dim, kernel_size=7)),
                             (f'act_fc{i}', act_f()),
                             (f'noise_fc{i}', noise_f()))


# Autoencoder ######################################################################################

class SimpleEncoder(Sequential):
    def __init__(self, kernel_sizes=(4,) * 4, widths=(32, 64, 128, 256), z_width=32,
                 norm_f=D.norm_f, act_f=nn.ReLU, conv_f=D.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if norm_f is not None:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', Linear(z_width))


# Adversarial autoencoder ##########################################################################
# TODO: linear to batchnorm, output shape

class AAEEncoder(Sequential):
    def __init__(self, kernel_sizes=(4,) * 3, widths=(64,) + (256,) * 2, z_dim=128,
                 norm_f=D.norm_f, act_f=nn.LeakyReLU, conv_f=D.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if i > 0:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', Linear(z_dim))


class AAEDecoder(Sequential):
    def __init__(self, h_dim=1024, kernel_sizes=(4,) * 3, widths=(256, 128, 1),
                 norm_f=D.norm_f, act_f=nn.ReLU, convt_f=D.convt_f):
        super().__init__()
        self.add_module('linear_h', Linear(h_dim))
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module({f'norm{i}': norm_f(),
                             f'act{i}': act_f(),
                             f'conv{i}': convt_f(kernel_size=k, out_channels=w, stride=2,
                                                 padding=1)})
        self.add_module('tanh', nn.Tanh())


class AAEDiscriminator(Sequential):
    def __init__(self, h_dim=default_args(AAEDecoder).h_dim, norm_f=D.norm_f, act_f=nn.ReLU):
        super().__init__()
        for i in range(2):
            # batch normalization?
            self.add_module(f'linear{i}', Linear(h_dim))
            self.add_module(f'act{i}', act_f())
        self.add_module('logits', Linear(2))
        self.add_module('probs', nn.Softmax(dim=1))
