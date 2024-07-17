from functools import partial, partialmethod
import functools
from fractions import Fraction as Frac
import typing as T
from warnings import warn
import logging
import sys

import torch
from typeguard import typechecked

import vidlu.modules as M
import vidlu.modules as vm
import vidlu.modules.components as vmc
from vidlu.modules.other import mnistnet, convnext
from vidlu.models.utils import ladder_input_names, set_all_inplace
from vidlu.utils.func import (Reserved, Empty, default_args)

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
                       dim_change='proj',
                       groups_f=vmc.ResNetV1Groups):
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
                  dim_change=dim_change, groups_f=groups_f)
    if isinstance(backbone_f, functools.partial) \
            and not len(inters := set(backbone_f.keywords).intersection(kwargs)) == 0:
        raise RuntimeError(f"Arguments {inters} should be given directly to the factory instead of "
                           f"being bound to backbone_f.")
    module = backbone_f(**kwargs)
    module.depth = depth
    return module


resnet_v2_backbone = partial(resnet_v1_backbone,
                             block_f=partial(default_args(vmc.ResNetV2Backbone).block_f,
                                             kernel_sizes=Reserved),
                             backbone_f=vmc.ResNetV2Backbone, groups_f=vmc.ResNetV2Groups)


def wide_resnet_backbone(depth, width_factor, small_input, dim_change='proj',
                         block_f=default_args(resnet_v2_backbone).block_f,
                         groups_f=vmc.ResNetV2Groups):
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
                                dim_change=dim_change,
                                groups_f=groups_f)


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


def convnext_backbone(name=None, depths=None, dims=None, backbone_f=convnext.ConvNeXtBackbone,
                      weights=None, **kwargs):
    if name is None:
        if depths is None or dims is None:
            raise RuntimeError("Either size or depths and dims should be provided.")
    elif name == 'tiny':
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
    else:
        raise ValueError(f"Invalid argument value: {name=}")
    result = backbone_f(depths=depths, dims=dims, **kwargs)
    if weights is not None:
        url = convnext.model_urls[weights]
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location='cpu',
                                                        check_hash=True)
        state_dict = {k: v for k, v in state_dict['model'].items()
                      if not k.startswith('norm') and not k.startswith('head')}
        result.load_state_dict(state_dict)
    return result


# Models ###########################################################################################

class Model(M.Module):
    def __init__(self, init=None):
        super().__init__()
        self._init = init or (lambda module: None)

    def initialize(self, input=None):
        if input is not None:
            self(input)
        if self._init is not None:
            self._init(self)


class DummyModel(Model):
    shape_to_output = dict()

    def __init__(self, model_f, *args, **kwargs):
        super().__init__()
        self.model = model_f(*args, **kwargs)

    def forward(self, x):
        if (result := self.shape_to_output.get(x.shape, None)) is None:
            with torch.no_grad():
                result = self.model(x)
                result = result[:1].expand_as(result)
                self.shape_to_output[x.shape] = result
        return result.to(x.device).detach().requires_grad_()


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
        if head_f is None:
            head_f = vm.Identity
        super().__init__(seq=dict(backbone=backbone_f(), head=head_f()), init=init,
                         input_adapter=input_adapter)


class ClassificationModel(DiscriminativeModel):
    __init__ = partialmethod(DiscriminativeModel.__init__,
                             head_f=vmc.ChannelAveragingClassificationHead)


class LogisticRegression(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=partial(M.Reshape, (-1, 1, 1)),
                             init=initialization.kaiming_resnet)


class SegmentationModel(DiscriminativeModel):
    def __init__(self, backbone_f, head_f, init, input_adapter=None, size_divisibility=None):
        super().__init__(backbone_f=backbone_f, head_f=head_f, init=init,
                         input_adapter=input_adapter)
        self.size_divisibility = size_divisibility

    def forward(self, x, shape=None):
        inject_shape = lambda m, h: (h[0], x.shape[-2:] if shape is None else shape)
        with self.head.register_forward_pre_hook(inject_shape):
            return super().forward(x)


class ResNetV1(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=partial(resnet_v1_backbone, base_width=64),
                             init=initialization.kaiming_resnet,
                             head_f=vmc.ChannelAveragingClassificationHead)


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
                             init=initialization.kaiming_densenet)


class IRevNet(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__, backbone_f=irevnet_backbone,
                             init=initialization.kaiming_resnet)

    def post_build(self, *args, **kwargs):
        super().post_build()
        for name, module in self.named_modules():
            if hasattr(module, 'inplace'):
                module.inplace = True  # ResNet-10: 8312MiB, 6.30/s -> 6734MiB, 6.32/s
        return True


class MNISTNet(ClassificationModel):
    __init__ = partialmethod(ClassificationModel.__init__,
                             backbone_f=mnistnet.MNISTNetBackbone,
                             init=initialization.kaiming_mnistnet,
                             head_f=vmc.heads.ClassificationHead1D)


swiftnet_ladder_f = partial(
    vmc.KresoLadderModel,
    context_f=partial(vmc.DenseSPP, bottleneck_size=128, level_size=42, out_size=128,
                      grid_sizes=(8, 4, 2)),
    lateral_preprocessing=lambda x: (torch.cat(x, dim=1) if isinstance(x, tuple) else x))
swiftnet_head_f = partial(vmc.heads.SegmentationHead, kernel_size=1)


@typechecked
class SwiftNetBase(SegmentationModel):
    def __init__(self,
                 backbone_f=resnet_v1_backbone,
                 up_width=128,
                 head_f=swiftnet_head_f,
                 input_adapter=None,
                 init=initialization.kaiming_resnet,
                 laterals=ladder_input_names,
                 # list(f"bulk.unit{i}_{j}" for i, j in zip(range(3), [1] * 3)),
                 lateral_suffix: T.Literal['sum', 'act', ''] = '',
                 stage_count=None,
                 ladder_f=swiftnet_ladder_f,
                 mem_efficiency=1):
        super().__init__(backbone_f=backbone_f, head_f=head_f, init=init,
                         input_adapter=input_adapter)
        self.laterals = laterals
        self.lateral_suffix = lateral_suffix
        self.mem_efficiency = mem_efficiency
        self.up_width = up_width
        self.stage_count = stage_count
        self.ladder_f = ladder_f

    @property
    def bulk(self):
        return self.backbone

    def build(self, x):
        if callable(self.laterals):
            vm.call_if_not_built(self.backbone, x)
            self.laterals = self.laterals(self.backbone)
        if self.stage_count is not None:
            self.laterals = self.laterals[-self.stage_count:]
            if self.stage_count != len(self.laterals):
                warn(f"{self.stage_count=} is different from {len(self.laterals)=}.")

        backbone = self.backbone  # self.backbone will be the decoder with the backbone (upsampling)
        self.backbone = self.ladder_f(
            backbone_f=lambda: backbone,
            laterals=[f"{p}.{self.lateral_suffix}" if self.lateral_suffix else
                      p for p in self.laterals],
            up_width=self.up_width,
            up_blend_f=partial(vmc.LadderUpsampleBlend, pre_blending='sum'),
            post_activation=True)
        super().build(x)


def swiftnet_set_mem_efficiency(model, mem_efficiency):
    set_all_inplace(model, mem_efficiency >= 1)
    if model.lateral_suffix == 'sum':
        for lb in model.laterals:
            vm.get_submodule(model.backbone.backbone, f"{lb}.act").inplace = False

    if mem_efficiency >= 3:  # 6022MiB 5.83/s
        for res_unit in model.backbone.backbone.bulk:
            res_unit.fork.block.set_checkpoints(('conv0', 'norm1'))
    elif mem_efficiency >= 2:  # 6260MiB, 5.84/s
        for res_unit in model.backbone.backbone.bulk:
            res_unit.fork.block.set_checkpoints(('conv0', 'act0'), ('conv1', 'norm1'))


class SwiftNet(SwiftNetBase):
    def post_build(self, *args, **kwargs):
        """Sets up in-place operations and gradient checkpointing for
        efficiency."""
        swiftnet_set_mem_efficiency(self, self.mem_efficiency)
        return True


class SwiftNetPyr(SwiftNetBase):
    __init__ = partialmethod(SwiftNetBase.__init__,
                             ladder_f=partial(vmc.MorsicPyramidModel, stage_count=3))
    post_build = SwiftNet.post_build


class SwiftNetIRevNet(SwiftNetBase):
    __init__ = partialmethod(SwiftNet.__init__,
                             backbone_f=partial(irevnet_backbone, no_final_postact=True))

    def post_build(self, *args, **kwargs):
        super().post_build()
        set_all_inplace(self, self.mem_efficiency >= 1)
        if self.mem_efficiency >= 2:
            raise NotImplementedError()
        return True


class SwiftNetConvNeXt(SwiftNetBase):
    __init__ = partialmethod(SwiftNet.__init__, backbone_f=convnext_backbone, init=None,
                             laterals=('stages.0', 'stages.1', 'stages.2'), mem_efficiency=0)

    def post_build(self, *args, **kwargs):
        super().post_build()
        if self.mem_efficiency > 0:
            raise NotImplementedError()
        return True


class LadderDensenet(DiscriminativeModel):
    def __init__(self, backbone_f=partial(densenet_backbone), laterals=None,
                 up_width=128, head_f=Empty, input_adapter=None):
        """

        laterals contains all but the last block?

        Args:
            backbone_f:
            laterals:
            up_width:
        """
        if laterals is None:
            laterals = tuple(f"bulk.db{i}.unit{j}.sum" for i, j in zip(range(3), [1] * 3))
            # TODO: automatic based on backbone, split DB3
        super().__init__(backbone_f=partial(vmc.KresoLadderModel,
                                            backbone_f=backbone_f,
                                            laterals=laterals,
                                            up_width=up_width),
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
