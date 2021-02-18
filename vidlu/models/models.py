from functools import partial, partialmethod
import functools
from fractions import Fraction as Frac
import typing as T
import warnings
import logging

import torch

import vidlu.modules as M
import vidlu.modules as vm
import vidlu.modules.components as vmc
from vidlu.modules.other import mnistnet
from vidlu.models.utils import ladder_input_names, set_inplace
from vidlu.utils.func import (Reserved, Empty, default_args)
from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj
from torch.utils import hooks

from . import initialization

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Backbones ########################################################################################


def resnet_v1_backbone(depth, base_width=default_args(vmc.ResNetV1Backbone).base_width,
                       small_input=default_args(vmc.ResNetV1Backbone).small_input,
                       block_f=partial(default_args(vmc.ResNetV1Backbone).block_f,
                                       kernel_sizes=Reserved),
                       backbone_f=vmc.ResNetV1Backbone,
                       group_lengths=(2,) * 4, width_factors=(1, 1), ksizes=(3, 3),
                       dim_change='proj'):
    # TODO: dropout
    if depth is not None:
        basic = ([3, 3], [1, 1], 'proj')  # maybe it should be 'pad' instead of 'proj'
        bottleneck = ([1, 3, 1], [1, 1, 4], 'proj')  # last paragraph in [2]
        group_lengths, (ksizes, width_factors, dim_change) = {
            10: ([1] * 4, basic),  # [1] bw 64
            18: ([2] * 4, basic),  # [1] bw 64
            34: ([3, 4, 6, 3], basic),  # [1] bw 64
            110: ([18] * 3, basic),  # [1] bw 16
            50: ([3, 4, 6, 3], bottleneck),  # [1] bw 64
            101: ([3, 4, 23, 3], bottleneck),  # [1] bw 64
            152: ([3, 8, 36, 3], bottleneck),  # [1] bw 64
            164: ([18] * 3, bottleneck),  # [1] bw 16
            200: ([3, 24, 36, 3], bottleneck),  # [2] bw 64
        }[depth]
    kwargs = dict(base_width=base_width, small_input=small_input, group_lengths=group_lengths,
                  block_f=partial(block_f, kernel_sizes=ksizes, width_factors=width_factors),
                  dim_change=dim_change)
    if isinstance(backbone_f, functools.partial) \
            and not len(inters := set(backbone_f.keywords).intersection(kwargs)) == 0:
        raise RuntimeError(f"Arguments {inters} should be given directly to the factory instead of "
                           f"being bound to backbone_f.")
    return backbone_f(**kwargs)


resnet_v2_backbone = partial(resnet_v1_backbone,
                             block_f=partial(default_args(vmc.ResNetV2Backbone).block_f,
                                             kernel_sizes=Reserved),
                             backbone_f=vmc.ResNetV2Backbone)


def wide_resnet_backbone(depth, width_factor, small_input, dim_change='proj',
                         block_f=default_args(resnet_v2_backbone).block_f):
    group_count, ksizes = 3, [3, 3]
    group_depth = (group_count * len(ksizes))
    zagoruyko_depth = depth
    blocks_per_group = (zagoruyko_depth - 4) // group_depth
    depth = blocks_per_group * group_depth + 4
    assert zagoruyko_depth == depth, \
        f"Invalid depth = {zagoruyko_depth} != {depth} = zagoruyko_depth"
    return vmc.ResNetV2Backbone(base_width=16,
                                small_input=small_input,
                                group_lengths=[blocks_per_group] * group_count,
                                block_f=partial(block_f, kernel_sizes=ksizes,
                                                width_factors=[width_factor] * 2),
                                dim_change=dim_change)


def densenet_backbone(depth, small_input, k=None, compression=0.5, ksizes=(1, 3),
                      block_f=partial(default_args(vmc.DenseNetBackbone).block_f,
                                      kernel_sizes=Reserved), backbone_f=vmc.DenseNetBackbone):
    # TODO: dropout 0.2
    # dropout if no pds augmentation
    depth_to_group_lengths = {
        121: ([6, 12, 24, 16], 32),
        161: ([6, 12, 36, 24], 48),
        169: ([6, 12, 32, 32], 32)}
    if depth in depth_to_group_lengths:
        db_lengths, default_growth_rate = depth_to_group_lengths[depth]
        k = k or default_growth_rate
    else:
        if k is None:
            raise ValueError("`k` (growth rate) must be supplied for non-Imagenet-model depth.")
        db_count = 3
        block_count = (depth - db_count - 1)
        if block_count % 3 != 0:
            raise ValueError(
                f"invalid depth: (depth-db_count-1) % 3 = {(depth - db_count - 1) % 3} != 0.")
        blocks_per_group = block_count // (db_count * len(ksizes))
        db_lengths = [blocks_per_group] * db_count
    return backbone_f(growth_rate=k,
                      small_input=small_input,
                      db_lengths=db_lengths,
                      compression=compression,
                      block_f=partial(block_f, kernel_sizes=ksizes))


mdensenet_backbone = partial(densenet_backbone,
                             block_f=partial(default_args(vmc.MDenseNetBackbone).block_f,
                                             kernel_sizes=Reserved),
                             backbone_f=vmc.MDenseNetBackbone)
fdensenet_backbone = partial(densenet_backbone,
                             block_f=partial(default_args(vmc.FDenseNetBackbone).block_f,
                                             kernel_sizes=Reserved),
                             backbone_f=vmc.FDenseNetBackbone)


def irevnet_backbone(init_stride=2, group_lengths=(6, 16, 72, 6),
                     block_f=partial(default_args(vmc.IRevNetBackbone).block_f,
                                     kernel_sizes=(3, 3, 3),
                                     width_factors=(Frac(1, 4), Frac(1, 4), 1)),
                     base_width=None,
                     groups_f=default_args(vmc.IRevNetBackbone).groups_f,
                     no_final_postact=False):
    return vmc.IRevNetBackbone(init_stride=init_stride, group_lengths=group_lengths,
                               base_width=base_width, block_f=block_f, groups_f=groups_f,
                               no_final_postact=no_final_postact)


# Models ###########################################################################################

class Model(M.Module):
    def __init__(self, init=None):
        super().__init__()
        self._init = init or (lambda module: None)

    def initialize(self, input=None):
        if input is not None:
            self(input)
        self._init(self)


class SeqModel(M.Seq):
    def __init__(self, seq, init, input_adapter=None):
        inpad = {} if input_adapter is None else dict(input_adapter=input_adapter)
        super().__init__(**inpad, **seq)
        self._init = init

    initialize = Model.initialize


class WrappedModel(SeqModel):
    def __init__(self, wrapped, init, input_adapter=None):
        super().__init__(seq=wrapped, init=init, input_adapter=input_adapter)


# Discriminative models ############################################################################

class DiscriminativeModel(SeqModel):
    def __init__(self, backbone_f, head_f, init, input_adapter=None):
        super().__init__(seq=dict(backbone=backbone_f(), head=head_f()), init=init,
                         input_adapter=input_adapter)


class ClassificationModel(DiscriminativeModel):
    pass


class LogisticRegression(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=partial(M.Reshape, (-1, 1, 1)),
                             init=initialization.kaiming_resnet)


class SegmentationModel(DiscriminativeModel):
    def forward(self, x, shape=None):
        inject_shape = lambda m, h: (h[0], x.shape[-2:] if shape is None else shape)
        with self.head.register_forward_pre_hook(inject_shape):
            return super().forward(x)


class ResNetV1(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=partial(resnet_v1_backbone, base_width=64),
                             init=initialization.kaiming_resnet)


class SegResNetV1(SegmentationModel):
    __init__ = ResNetV1.__init__

    def post_build(self, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        for unit_name, unit in self.backbone.bulk.named_children():
            self.backbone.bulk.set_modifiers(
                **{unit_name: lambda module: partial(checkpoint, module)})


class ResNetV2(ClassificationModel):
    __init__ = partialmethod(ResNetV1.__init__,
                             backbone_f=partial(resnet_v2_backbone, base_width=64))


class WideResNet(ResNetV2):
    __init__ = partialmethod(ResNetV2.__init__, backbone_f=wide_resnet_backbone)


WRN = WideResNet


class DenseNet(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=densenet_backbone,
                             init=partial(initialization.kaiming_densenet, module=Reserved))


class IRevNet(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__, backbone_f=irevnet_backbone,
                             init=initialization.kaiming_resnet)

    def post_build(self, *args, **kwargs):
        super().post_build()
        for name, module in self.named_modules():
            if hasattr(module, 'inplace'):
                module.inplace = True  # ResNet-10: 8312MiB, 6.30/s -> 6734MiB, 6.32/s


class MNISTNet(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=mnistnet.MNISTNetBackbone,
                             init=initialization.kaiming_mnistnet)


class SwiftNetBase(SegmentationModel):
    def __init__(self,
                 backbone_f=resnet_v1_backbone,
                 ladder_width=128,
                 head_f=vmc.heads.SegmentationHead,
                 input_adapter=None,
                 init=initialization.kaiming_resnet,
                 laterals=None,  # list(f"bulk.unit{i}_{j}" for i, j in zip(range(3), [1] * 3)),
                 lateral_suffix: T.Literal['sum', 'act', ''] = '',
                 mem_efficiency=1):
        if lateral_suffix not in ('sum', 'act', ''):
            raise ValueError("lateral_suffix should be either 'sum' or 'act'.")
        super().__init__(
            backbone_f=backbone_f,
            head_f=partial(head_f, kernel_size=1),
            init=init,
            input_adapter=input_adapter)
        self.laterals = laterals
        self.lateral_suffix = lateral_suffix
        self.mem_efficiency = mem_efficiency
        self.ladder_width = ladder_width

    def build(self, x):
        backbone = self.backbone
        if self.laterals is None or callable(self.laterals):
            backbone(x)
            self.laterals = self.laterals(backbone) if callable(self.laterals) else \
                ladder_input_names(backbone)
        self.backbone = vmc.KresoLadderNet(
            backbone_f=lambda: backbone,
            laterals=[f"{p}.{self.lateral_suffix}" if self.lateral_suffix else
                      p for p in self.laterals],
            ladder_width=self.ladder_width,
            context_f=partial(vmc.DenseSPP, bottleneck_size=128, level_size=42,
                              out_size=128, grid_sizes=(8, 4, 2)),
            up_blend_f=partial(vmc.LadderUpsampleBlend, pre_blending='sum'),
            post_activation=True,
            lateral_preprocessing=lambda x: (torch.cat(x, dim=1) if isinstance(x, tuple) else x))
        super().build(x)


class SwiftNet(SwiftNetBase):
    def post_build(self, *args, **kwargs):
        """Sets up in-place operations and gradient checkpointing for
        efficiency."""
        super().post_build()

        set_inplace(self, self.mem_efficiency >= 1)
        if self.lateral_suffix == 'sum':
            for lb in self.laterals:
                vm.get_submodule(self.backbone.backbone, f"{lb}.act").inplace = False

        if self.mem_efficiency >= 3:  # 6022MiB 5.83/s
            for res_unit in self.backbone.backbone.bulk:
                res_unit.fork.block.set_checkpoints(('conv0', 'norm1'))
        elif self.mem_efficiency >= 2:  # 6260MiB, 5.84/s
            for res_unit in self.backbone.backbone.bulk:
                res_unit.fork.block.set_checkpoints(('conv0', 'act0'), ('conv1', 'norm1'))


class SwiftNetIRevNet(SwiftNetBase):
    __init__ = partialmethod(SwiftNet.__init__,
                             backbone_f=partial(irevnet_backbone, no_final_postact=True))

    def post_build(self, *args, **kwargs):
        super().post_build()
        set_inplace(self, self.mem_efficiency >= 1)
        if self.mem_efficiency >= 2:
            raise NotImplementedError()


class LadderDensenet(DiscriminativeModel):
    def __init__(self, backbone_f=partial(densenet_backbone), laterals=None,
                 ladder_width=128, head_f=Empty, input_adapter=None):
        """

        laterals contains all but the last block?

        Args:
            backbone_f:
            laterals:
            ladder_width:
        """
        if laterals is None:
            laterals = tuple(f"bulk.db{i}.unit{j}.sum" for i, j in zip(range(3), [1] * 3))
            # TODO: automatic based on backbone, split DB3
        super().__init__(backbone_f=partial(vmc.KresoLadderNet,
                                            backbone_f=backbone_f,
                                            laterals=laterals,
                                            ladder_width=ladder_width),
                         head_f=head_f,
                         init=initialization.kaiming_resnet,
                         input_adapter=input_adapter)


class BackbonelessSegmentator(DiscriminativeModel):
    def __init__(self, head_f=Empty, input_adapter=None):
        super().__init__(backbone_f=vm.Identity,
                         head_f=head_f,
                         init=lambda m: m,
                         input_adapter=input_adapter)


# Autoencoders #####################################################################################

class Autoencoder(Model):
    def __init__(self, encoder_f, decoder_f, init):
        super().__init__(init=init)
        self.encoder, self.decoder = encoder_f(), decoder_f()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# Adversarial autoencoder

class AdversarialAutoencoder(Autoencoder):
    def __init__(self, encoder_f=vmc.AAEEncoder, decoder_f=vmc.AAEDecoder,
                 discriminator_f=vmc.AAEDiscriminator,
                 prior_rand_f=partial(torch.randn, std=0.3), init=None):
        super().__init__(encoder_f, decoder_f, init)
        self.discriminator = discriminator_f()
        self.prior_rand = prior_rand_f()

    def discriminate_z(self, z):
        return self.discriminator(z)


# GANs #############################################################################################

class GAN(Model):
    def __init__(self, generator_f, discriminator_f, z_shape, z_rand_f=torch.randn, init=Empty):
        super().__init__(init=init)
        self.z_shape, self.z_rand = z_shape, z_rand_f()
        self.generator, self.discriminator = generator_f(), discriminator_f()

    def sample_z(self, batch_size):
        self.z_rand(batch_size, *self.z_shape, device=self.device)


# Other models #####################################################################################

class SmallImageClassifier(Model):
    def __init__(self):
        super().__init__()
        from torch import nn
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
