from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .modules import (Module, Conv, ConvTranspose, Sequential, MaxPool, BatchNorm, AvgPool, Func,
                      Linear)
from vidlu.utils.func import (default_args, params, Empty, Reserved)
from vidlu.utils.misc import (clean_locals)
from vidlu.utils.collections import NameDict

# Default arguments ################################################################################

_d = NameDict(
    norm_f=BatchNorm,
    act_f=partial(nn.ReLU, inplace=True),
    conv_f=partial(Conv, padding='half', bias=False),
    convt_f=partial(ConvTranspose, bias=False),
)


# ResNet/DenseNet root block #######################################################################

class RootBlock(Sequential):
    def __init__(self, out_channels: int, small_input, norm_f=_d.norm_f, act_f=_d.act_f):
        if small_input:  # CIFAR
            super().__init__(conv=Conv(out_channels, 3, padding='half', bias=False))
        else:
            super().__init__(conv=Conv(out_channels, kernel_size=7, padding='half', bias=False),
                             norm=norm_f(), activation=act_f(), pool=MaxPool(3, stride=2))


# blocks ###########################################################################################

class PreactBlock(Sequential):
    # normalization -> activation [-> noise] -> convolution
    def __init__(self, kernel_sizes, widths, noise_locations=[], stride=1,
                 omit_first_preactivation=False, norm_f=_d.norm_f, act_f=_d.act_f,
                 conv_f=partial(_d.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved),
                 noise_f=None):
        super().__init__()
        add = self.add_module
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            if i > 0 or not omit_first_preactivation:
                add(f'norm{i}', norm_f())
                add(f'act{i}', act_f())
            if i in noise_locations:
                add(f'noise{i}', noise_f())
            add(f'conv{i}', conv_f(kernel_size=k, out_channels=w, stride=stride if i == 0 else 1))


class PostactBlock(Sequential):
    # convolution -> normalization -> activation [-> noise]
    def __init(self, kernel_sizes, widths, noise_locations=(), stride=1, omit_first_norm=False,
               norm_f=_d.norm_f, act_f=_d.act_f,
               conv_f=partial(_d.conv_f, kernel_size=Reserved, out_channels=Reserved,
                              stride=Reserved),
               noise_f=None):
        super().__init__()
        add = self.add_module
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            add(f'conv{i}', conv_f(kernel_size=k, out_channels=w, stride=stride if i == 0 else 1))
            if norm_f and i > 0 or not omit_first_norm:
                add(f'norm{i}', norm_f())
            add(f'act{i}', act_f())
            if i in noise_locations:
                add(f'noise{i}', noise_f())


# invertible #######################################################################################

class AffineCoupling(Module):
    def __init__(self, scale_f, translate_f):
        super().__init__()
        self.scale, self.translate = scale_f(), translate_f()

    def forward(self, x1, x2):
        x1, x1 * self.scale(x1).exp() + self.translate(x1)

    def invert(self, y1, y2):
        return y1, (y2 - self.translate(y1)) * (-self.scale(y1)).exp()


# heads ############################################################################################

class ClassificationHead(Sequential):
    def __init__(self, class_count):
        super().__init__(pre_logits_mean=nn.AdaptiveAvgPool2d((1, 1)),
                         logits=Linear(class_count),
                         probs=nn.Softmax(dim=1))


class SegmentationHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, align_corners=False)),
                         probs=nn.Softmax(dim=1))


class RegressionHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, align_corners=False)),
                         probs=nn.Softmax(dim=1))


# VGG ##############################################################################################

class VGGBackbone(Sequential):
    def __init__(self, base_width=64, block_depths=[2, 2, 3, 3, 3],
                 block_f=partial(PostactBlock, kernel_sizes=Reserved, widths=Reserved, norm_f=None),
                 pool_f=partial(MaxPool, ceil_mode=True)):
        super().__init__()
        add = self.add_module
        for i, d in enumerate(block_depths):
            add(f'block{i}', block_f(kernel_sizes=[3] * d, widths=[base_width * 2 ** i] * d))
            add(f'pool{i}', pool_f(kernel_size=2, stride=2))


class VGGClassifier(Sequential):
    def __init__(self, fc_dim, class_count, act_f=_d.act_f, noise_f=nn.Dropout):
        super().__init__()
        add = self.add_module
        widths = [fc_dim] * 2 + [class_count]
        for i, w in enumerate(widths):
            add(f'linear{i}', Linear(w))
            add(f'act_fc{i}', act_f())
            if noise_f:
                add(f'noise_fc{i}', noise_f())
        add('probs', nn.Softmax(dim=1))


# FCN ##############################################################################################

class FCNEncoder(Sequential):
    def __init__(self, block_f=default_args(VGGBackbone).block_f,
                 pool_f=partial(MaxPool, ceil_mode=True), fc_dim=4096, noise_f=nn.Dropout2d):
        super().__init__()
        add = self.add_module
        add('vgg_backbone', VGGBackbone(block_f=block_f))

        conv_f = default_args(block_f).conv_f
        act_f = default_args(block_f).act_f
        for i in range(2):
            add(f'conv_fc{i}', conv_f(fc_dim, kernel_size=7))
            add(f'act_fc{i}', act_f())
            add(f'noise_fc{i}', noise_f())


# ResNet ###########################################################################################

class ResUnit(Module):
    def __init__(self, block_f=partial(PreactBlock, omit_first_preactivation=Reserved),
                 dim_change='proj'):
        super().__init__()
        assert dim_change in ['pad', 'proj']
        assert all(params(block_f)[x] is not Empty for x in ['kernel_sizes', 'widths'])
        block = block_f(omit_first_preactivation=False)
        self.preact = block[:2]
        self.block = block[2:]
        del block
        self.shortcut = None

    def build(self, x):
        block_args = default_args(self.args.block_f)
        out_width, stride = block_args.widths[-1], block_args.stride
        if stride == 1 and x.shape[1] == out_width:
            self.shortcut = lambda x: x
        else:
            if self.args.dim_change == 'proj':
                self.shortcut = Conv(out_width, kernel_size=1, stride=stride, bias=False)
            else:
                pad = [0] * 5 + [out_width - x.shape[1]]
                self.shortcut = Sequential(pool=AvgPool(stride, stride),
                                           pad=Func(partial(F.pad, pad=pad, mode='constant')))

    def forward(self, x):
        p = self.preact(x)
        return self.block(p) + self.shortcut(p if self.args.dim_change == 'proj' else x)


class ResGroups(Sequential):
    def __init__(self, group_lengths, base_widths,
                 block_f=partial(default_args(ResUnit).block_f, widths=Reserved, stride=Reserved),
                 dim_change=default_args(ResUnit).dim_change):
        super().__init__()
        for i, l in enumerate(group_lengths):
            for j in range(l):
                self.add_module(f'unit{i}_{j}',
                                ResUnit(block_f=partial(block_f,
                                                        widths=[2 ** i * w for w in base_widths],
                                                        stride=1 + int(i > 0 and j == 0)),
                                        dim_change=dim_change))
        self.add_module('post_norm', default_args(block_f).norm_f())
        self.add_module('post_act', default_args(block_f).act_f())


class ResNetBackbone(Sequential):
    def __init__(self, base_width=16, small_input=True, group_lengths=[2] * 4,
                 width_factors=[16] * 2, block_f=default_args(ResGroups).block_f,
                 dim_change=default_args(ResGroups).dim_change):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(
            root=RootBlock(base_width, small_input, **norm_act_args),
            features=ResGroups(group_lengths, [base_width * wf for wf in width_factors], block_f,
                               dim_change))

    def forward(self, x):
        return super().forward(x)


# DenseNet #########################################################################################

class DenseTransition(Sequential):
    def __init__(self, compression=0.5, norm_f=_d.norm_f, act_f=_d.act_f,
                 conv_f=partial(_d.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.args = NameDict(clean_locals())

    def build(self, x):
        self.add_module(norm=self.args.norm_f())
        self.add_module(act=self.args.act_f())
        if self.args.noise_f is not None:
            self.add_module(noise=self.args.noise_f())
        self.add_module(conv=self.args.conv_f(out_features=round(x.shape[1] * self.args.compression)))
        self.add_module(pool=partial(self.args.pool_f, stride=2))


class DenseUnit(Module):
    def __init__(self, block_f=partial(PreactBlock, kernel_sizes=[1, 3], widths=[4 * 12, 12])):
        super().__init__()
        self.block = block_f()

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class DenseBlock(Sequential):
    def __init__(self, length, block_f=default_args(DenseUnit).block_f):
        super().__init__({f'unit{i}': DenseUnit(block_f) for i in range(length)})


class DenseSequence(Sequential):
    def __init__(self, db_lengths=[2] * 4, compression=default_args(DenseTransition).compression,
                 block_f=default_args(DenseBlock).block_f):
        super().__init__()
        add = self.add_module
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        for i, l in enumerate(db_lengths):
            if i > 0:
                add(f'transition{i - 1}', DenseTransition(compression, **norm_act_args))
            add(f'dense_block{i}', DenseBlock(l, block_f))


class DenseNetBackbone(Sequential):
    def __init__(self, base_width=12, small_input=True, db_lengths=[2] * 4,
                 compression=default_args(DenseSequence).compression,
                 block_f=default_args(DenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=RootBlock(2 * base_width, small_input, **norm_act_args),
                         features=DenseSequence(db_lengths, compression, block_f))


# Experimental #####################################################################################

class SlugUnit(Module):
    def __init__(self, change_dim=False,
                 block_f=partial(PreactBlock, kernel_sizes=[1, 3], width_factors=[4, 1])):
        super().__init__()
        self.blocks = None

    def build(self, inputs):
        last_dim = inputs[-1].shape[2]
        stride_factor = 1 + int(self.args.change_dim)
        strides = [stride_factor * (1 + int(x.shape[2] > last_dim)) for x in inputs]
        self.blocks = nn.ModuleList([self.args.block_f(stride=s) for s in strides])

    def forward(self, inputs):
        return sum(b(x) for b, x in zip(self.blocks, inputs))


# Autoencoder ######################################################################################

class SimpleEncoder(Sequential):
    def __init__(self, kernel_sizes=(4,) * 4, widths=(32, 64, 128, 256), z_width=32,
                 norm_f=_d.norm_f, act_f=nn.ReLU, conv_f=_d.conv_f):
        super().__init__()
        add = self.add_module
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            add(f'conv{i}', conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if norm_f is not None:
                add(f'norm{i}', norm_f())
            add(f'act{i}', act_f())
        add('linear_z', Linear(z_width))


# Adversarial autoencoder ##########################################################################


class AAEEncoder(Sequential):
    def __init__(self, kernel_sizes=(4,) * 3, widths=(64,) + (256,) * 2, z_dim=128,
                 norm_f=_d.norm_f, act_f=nn.LeakyReLU, conv_f=_d.conv_f):
        super().__init__()
        add = self.add_module
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            add(f'conv{i}', conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if i > 0:
                add(f'norm{i}', norm_f())
            add(f'act{i}', act_f())
        add('linear_z', Linear(z_dim))


# TODO: linear to batchnorm, output shape
class AAEDecoder(Sequential):
    def __init__(self, h_dim=1024, kernel_sizes=(4,) * 3, widths=(256, 128, 1),
                 norm_f=_d.norm_f, act_f=nn.ReLU, convt_f=_d.convt_f):
        super().__init__()
        add = self.add_module
        add('linear_h', Linear(h_dim))
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            add(f'norm{i}', norm_f())
            add(f'act{i}', act_f())
            add(f'conv{i}', convt_f(kernel_size=k, out_channels=w, stride=2, padding=1))
        add('tanh', nn.Tanh())


class AAEDiscriminator(Sequential):
    def __init__(self, h_dim=default_args(AAEDecoder).h_dim, norm_f=_d.norm_f, act_f=nn.ReLU):
        super().__init__()
        add = self.add_module
        for i in range(2):
            # batch normalization?
            add(f'linear{i}', Linear(h_dim))
            add(f'act{i}', act_f())
        add('logits', Linear(2))
        add('probs', nn.Softmax(dim=1))
