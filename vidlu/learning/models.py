from functools import partial, partialmethod
from abc import ABC

from torch import nn

from vidlu.nn.modules import Sequential
from vidlu.nn.components import ResNetBackbone, DenseNetBackbone, classification_head
from vidlu.nn import init
from vidlu.utils.func import ArgTree, argtree_partialmethod


# Feature extractors ###############################################################################

def resnet_backbone(depth, base_width, small_input, _f=ResNetBackbone):
    # TODO: dropout
    print(f'ResNet-{depth}-{base_width}')
    basic = ([3, 3], [1, 1], 'pad')
    bottleneck = ([1, 3, 1], [1, 1, 4], 'proj')  # last paragraph in [2]
    group_lengths, (ksizes, width_factors, dim_change) = {
        18: ([2] * 4, basic),  # [1] bw 64
        34: ([3, 4, 6, 3], basic),  # [1] bw 64
        110: ([18] * 3, basic),  # [1] bw 16
        50: ([3, 4, 6, 3], bottleneck),  # [1] bw 64
        101: ([3, 4, 23, 3], bottleneck),  # [1] bw 64
        152: ([3, 8, 36, 3], bottleneck),  # [1] bw 64
        164: ([18] * 3, bottleneck),  # [1] bw 16
        200: ([3, 24, 36, 3], bottleneck),  # [2] bw 64
    }[depth]
    return _f(base_width=base_width,
              small_input=small_input,
              group_lengths=group_lengths,
              width_factors=width_factors,
              block_f=partial(ResNetBackbone.block_f, kernel_sizes=ksizes),
              dim_change=dim_change)


def wide_resnet_backbone(depth, width_factor, small_input, dropout, dim_change, _f=ResNetBackbone):
    print(f'WRN-{depth}-{width_factor}')
    zagoruyko_depth = depth

    group_count, ksizes = 3, [3, 3]
    group_depth = (group_count * len(ksizes))
    blocks_per_group = (zagoruyko_depth - 4) // group_depth
    depth = blocks_per_group * group_depth + 4
    assert zagoruyko_depth == depth, \
        f"Invalid depth = {zagoruyko_depth} != {depth} = zagoruyko_depth"

    d = {}
    if dropout:
        d['p'] = dropout if type(dropout) in [int, float] else 0.3
    noise_f = partial(_f.block_f.noise_f, **d)

    return _f(base_width=16,
              small_input=small_input,
              group_lengths=[blocks_per_group] * group_count,
              width_factors=[width_factor] * 2,
              block_f=partial(_f.block_f,
                              kernel_sizes=ksizes,
                              noise_locations=[0] if dropout else [],
                              noise_f=noise_f),
              dim_change=dim_change)


def densenet_backbone(depth, base_width, small_input, compression=0.5, _f=DenseNetBackbone):
    # TODO: dropout 0.2
    # dropout if no data augmentation
    print(f'DenseNet-{depth}-{base_width}' if base_width else f'DenseNet-{depth}')
    ksizes = [1, 3]
    depth_to_group_lengths = {
        121: ([6, 12, 24, 16], 32),
        161: ([6, 12, 36, 24], 48),
        169: ([6, 12, 32, 32], 32),
    }
    if depth in depth_to_group_lengths:
        db_lengths, bw = depth_to_group_lengths[depth]
        base_width = base_width or bw
    else:
        assert base_width is not None, "base_width not supplied for non-standard depth"
        db_count = 3
        assert (depth - db_count - 1) % 3 == 0, \
            f"invalid depth: (depth-group_count-1) must be divisible by 3"
        blocks_per_group = (depth - db_count - 1) // (db_count * len(ksizes))
        db_lengths = [blocks_per_group] * db_count
    return _f(base_width=base_width,
              small_input=small_input,
              db_lengths=db_lengths,
              compression=compression,
              block_f=partial(DenseNetBackbone.block_f, kernel_sizes=ksizes))


# Models ###########################################################################################

class Model(nn.Module, ABC):
    def __init__(self, init):
        super().__init__()
        self._init = init

    def initialize_parameters(self):
        self._init(self)


# Discriminative

class DiscriminativeModel(Sequential, Model):
    def __init__(self, backbone_f, head_f, init):
        Sequential.__init__(self, fex=backbone_f(), head=head_f())
        Model.__init__(self, init=init)


class ResNet(DiscriminativeModel):
    __init__ = partialmethod(DiscriminativeModel.__init__, backbone_f=resnet_backbone,
                             init=init.resnet)


class WideResNet(ResNet):
    __init__ = partialmethod(ResNet.__init__, backbone_f=wide_resnet_backbone)


class DenseNet(DiscriminativeModel):
    __init__ = partialmethod(DiscriminativeModel.__init__, backbone_f=densenet_backbone,
                             init=init.densenet)


# Variants for the purpose of shorter names

class ResNet18(ResNet):
    __init__ = argtree_partialmethod(ResNet.__init__, backbone_f=ArgTree(depth=18, base_width=64))


# Autoencoder

class Autoencoder(Sequential, Model):
    def __init__(self, encoder_f, decoder_f, init):
        Sequential.__init__(self, encoder=encoder_f(), decoder=decoder_f())
        Model.__init__(self, init=init)


# GAN

class GAN(Sequential, Model):
    def __init__(self, generator_f, discriminator_f, init):
        Sequential.__init__(self, generator=generator_f(), discriminator=discriminator_f())
        Model.__init__(self, init=init)
