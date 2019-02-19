from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .modules import (Module, Conv, ConvTranspose, Sequential, MaxPool, BatchNorm, AvgPool, Func,
                      Linear, Func)
from vidlu.utils.func import (do, pipe, default_args, params, Empty, Reserved)
from vidlu.utils.misc import (locals_c)
from vidlu.utils.collections import NameDict

# Default arguments ################################################################################

_d = NameDict(
    norm_f=BatchNorm,
    act_f=partial(nn.ReLU, inplace=False),  # TODO: Can "inplace=True" cause bugs?
    conv_f=partial(Conv, padding='half', bias=False),
    convt_f=partial(ConvTranspose, bias=False),
)


def _call_no_inplace(module_f):
    if 'inplace' in params(module_f):
        return module_f(inplace=False)
    return module_f()


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
    def __init__(self, kernel_sizes, base_width, width_factors, noise_locations=(), stride=1,
                 omit_first_preactivation=False,
                 norm_f=_d.norm_f, act_f=_d.act_f,
                 conv_f=partial(_d.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved),
                 noise_f=None):
        super().__init__()
        widths = [base_width * wf for wf in width_factors]
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            if i > 0 or not omit_first_preactivation:
                if norm_f:
                    self.add_module(f'norm{i}', norm_f())
                self.add_module(f'act{i}', act_f())
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=stride if i == 0 else 1))


class PostactBlock(Sequential):
    # convolution -> normalization -> activation [-> noise]
    def __init__(self, kernel_sizes, base_width, width_factors, noise_locations=(), stride=1,
                 norm_f=_d.norm_f, act_f=_d.act_f,
                 conv_f=partial(_d.conv_f, kernel_size=Reserved, out_channels=Reserved,
                                stride=Reserved),
                 noise_f=None):
        super().__init__()
        widths = [base_width * wf for wf in width_factors]
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}', Reserved.call(conv_f, kernel_size=k, out_channels=w,
                                                      stride=stride if i == 0 else 1))
            if norm_f:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
            if i in noise_locations:
                self.add_module(f'noise{i}', noise_f())


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
                         log_probs=nn.LogSoftmax(dim=1))


class SegmentationHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, align_corners=False)),
                         log_probs=nn.LogSoftmax(dim=1))


class RegressionHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, align_corners=False)),
                         log_probs=nn.LogSoftmax(dim=1))


# VGG ##############################################################################################

class VGGBackbone(Sequential):
    def __init__(self, base_width=64, block_depths=(2, 2, 3, 3, 3),
                 block_f=partial(PostactBlock, kernel_sizes=Reserved, base_width=Reserved,
                                 width_factors=Reserved, norm_f=None),
                 pool_f=partial(MaxPool, ceil_mode=True)):
        super().__init__()
        for i, d in enumerate(block_depths):
            self.add_module(f'block{i}',
                            Reserved.call(block_f, kernel_sizes=[3] * d, base_width=base_width,
                                          width_factors=[2 ** i] * d))
            self.add_module(f'pool{i}', pool_f(kernel_size=2, stride=2))


class VGGClassifier(Sequential):
    def __init__(self, fc_dim, class_count, act_f=_d.act_f, noise_f=nn.Dropout):
        super().__init__()
        widths = [fc_dim] * 2 + [class_count]
        for i, w in enumerate(widths):
            self.add_module(f'linear{i}', Linear(w))
            self.add_module(f'act_fc{i}', act_f())
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
            self.add_module(f'conv_fc{i}', conv_f(fc_dim, kernel_size=7))
            self.add_module(f'act_fc{i}', act_f())
            self.add_module(f'noise_fc{i}', noise_f())


# ResNet ###########################################################################################

def _check_block_args(block_f):
    args = params(block_f)
    if 'kernel_sizes' in args and args['kernel_sizes'] is Empty:
        raise ValueError("Argument kernel_sizes missing in block_f.")


class ResUnit(Module):
    def __init__(self, block_f=partial(PreactBlock, omit_first_preactivation=Reserved),
                 dim_change='proj'):
        _check_block_args(block_f)
        super().__init__()
        if dim_change not in ['pad', 'proj']:
            raise ValueError(f"Invalid value for argument dim_change: {dim_change}.")
        block = Reserved.call(block_f, omit_first_preactivation=False)
        self.preact = block[:'conv0']
        self.block = block['conv0':]
        del block
        self.shortcut = None

    def build(self, x):
        block_args = default_args(self.args.block_f)
        out_width = block_args.base_width * block_args.width_factors[-1]
        stride = block_args.stride
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
    def __init__(self, group_lengths, base_width, width_factors,
                 block_f=partial(default_args(ResUnit).block_f, base_width=Reserved,
                                 width_factors=Reserved, stride=Reserved),
                 dim_change=default_args(ResUnit).dim_change):
        super().__init__()
        _check_block_args(block_f)
        for i, l in enumerate(group_lengths):
            for j in range(l):
                self.add_module(f'unit{i}_{j}',
                                ResUnit(block_f=Reserved.partial(block_f,
                                                                 base_width=base_width * 2 ** i,
                                                                 width_factors=width_factors,
                                                                 stride=1 + int(i > 0 and j == 0)),
                                        dim_change=dim_change))
        norm_f = default_args(block_f).norm_f
        if norm_f is not None:
            self.add_module('post_norm', norm_f())
        self.add_module('post_act', default_args(block_f).act_f())


class ResNetBackbone(Sequential):
    def __init__(self, base_width=64, small_input=True, group_lengths=(2,) * 4,
                 width_factors=(1,) * 2, block_f=default_args(ResGroups).block_f,
                 dim_change=default_args(ResGroups).dim_change):
        _check_block_args(block_f)
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(
            root=RootBlock(base_width, small_input, **norm_act_args),
            features=ResGroups(group_lengths, base_width=base_width, width_factors=width_factors,
                               block_f=block_f, dim_change=dim_change))


# DenseNet #########################################################################################

class DenseTransition(Sequential):
    def __init__(self, compression=0.5, norm_f=_d.norm_f, act_f=_d.act_f,
                 conv_f=partial(_d.conv_f, kernel_size=1), noise_f=None,
                 pool_f=partial(AvgPool, kernel_size=2, stride=Reserved)):
        super().__init__()
        self.args = NameDict(locals_c())

    def build(self, x):
        self.add_module(norm=self.args.norm_f())
        self.add_module(act=self.args.act_f())
        if self.args.noise_f is not None:
            self.add_module(noise=self.args.noise_f())
        self.add_module(
            conv=self.args.conv_f(out_features=round(x.shape[1] * self.args.compression)))
        self.add_module(pool=Reserved.call(self.args.pool_f, stride=2))


class DenseUnit(Module):
    def __init__(self, block_f=partial(PreactBlock, kernel_sizes=(1, 3), width_factors=(4, 1))):
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
        for i, len in enumerate(db_lengths):
            if i > 0:
                self.add_module(f'transition{i - 1}', DenseTransition(compression, **norm_act_args))
            self.add_module(f'dense_block{i}',
                            DenseBlock(len,
                                       block_f=Reserved.partial(block_f, base_width=growth_rate)))


class DenseNetBackbone(Sequential):
    def __init__(self, growth_rate=12, small_input=True, db_lengths=(2,) * 4,
                 compression=default_args(DenseSequence).compression,
                 block_f=default_args(DenseSequence).block_f):
        norm_act_args = {k: default_args(block_f)[k] for k in ['norm_f', 'act_f']}
        super().__init__(root=RootBlock(2 * growth_rate, small_input, **norm_act_args),
                         features=DenseSequence(growth_rate, db_lengths, compression, block_f))


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
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if norm_f is not None:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', Linear(z_width))


# Adversarial autoencoder ##########################################################################


class AAEEncoder(Sequential):
    def __init__(self, kernel_sizes=(4,) * 3, widths=(64,) + (256,) * 2, z_dim=128,
                 norm_f=_d.norm_f, act_f=nn.LeakyReLU, conv_f=_d.conv_f):
        super().__init__()
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'conv{i}',
                            conv_f(kernel_size=k, out_channels=w, stride=2, bias=i == 0))
            if i > 0:
                self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
        self.add_module('linear_z', Linear(z_dim))


# TODO: linear to batchnorm, output shape
class AAEDecoder(Sequential):
    def __init__(self, h_dim=1024, kernel_sizes=(4,) * 3, widths=(256, 128, 1),
                 norm_f=_d.norm_f, act_f=nn.ReLU, convt_f=_d.convt_f):
        super().__init__()
        self.add_module('linear_h', Linear(h_dim))
        for i, (k, w) in enumerate(zip(kernel_sizes, widths)):
            self.add_module(f'norm{i}', norm_f())
            self.add_module(f'act{i}', act_f())
            self.add_module(f'conv{i}', convt_f(kernel_size=k, out_channels=w, stride=2, padding=1))
        self.add_module('tanh', nn.Tanh())


class AAEDiscriminator(Sequential):
    def __init__(self, h_dim=default_args(AAEDecoder).h_dim, norm_f=_d.norm_f, act_f=nn.ReLU):
        super().__init__()
        for i in range(2):
            # batch normalization?
            self.add_module(f'linear{i}', Linear(h_dim))
            self.add_module(f'act{i}', act_f())
        self.add_module('logits', Linear(2))
        self.add_module('probs', nn.Softmax(dim=1))


# Torch resnet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
