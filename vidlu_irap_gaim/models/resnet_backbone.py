"""
Exact port of libs/irap_gaim-main/resnet.py for IRAP GAIM compatibility.
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.nn.functional as F
import warnings

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_class = nn.Conv2d
    return conv_class(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return bn_function


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)

        out = bn_1(x)
        out = bn_2(out)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, separable=False):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        conv_class = nn.Conv2d
        self.conv2 = conv_class(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3, self.relu)

        out = bn_1(x)
        out = bn_2(out)
        out = bn_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0):
        super(_BNReluConv, self).__init__()

        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))

        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))

        padding = k // 2

        conv_class = nn.Conv2d

        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias, dilation=dilation))
        if drop_rate > 0:
            warnings.warn(f'Using dropout with p: {drop_rate}')
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling, self).__init__()

        self.fixed_size = fixed_size
        self.grids = grids

        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))

        self.square_grid = square_grid

        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn and starts_with_bn))

        num_features = bt_size
        final_size = num_features

        for i in range(len(grids)):
            final_size += level_size
            # Note: hardcoded 128->42 as in original line 280
            self.spp.add_module('spp' + str(i), _BNReluConv(128, 42, k=1, bn_momentum=bn_momentum, batch_norm=use_bn, drop_rate=drop_rate))

    def forward(self, x):
        N = x.shape[0]

        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        num = len(self.spp)

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])

            level = self.spp[i].forward(x_pooled)

            level = level.view(N, -1)

            levels.append(level)

        x = torch.cat(levels, 1)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, use_bn=True,
                 spp_grids=(6, 3, 2, 1), spp_square_grid=True, spp_drop_rate=0.0,
                 target_size=None, output_stride=4,
                 **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.target_size = target_size

        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 6)]
        else:
            target_sizes = [None] * 4

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]

        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3

        self.spp_size = kwargs.get('spp_size', num_features)
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features, grids=spp_grids, square_grid=True,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, drop_rate=spp_drop_rate,
                                         fixed_size=target_sizes[3])

        self.random_init = [self.spp]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def rn_backbone(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.forward_resblock(x, self.layer1)
        x, _ = self.forward_resblock(x, self.layer2)
        x, _ = self.forward_resblock(x, self.layer3)
        x, skip = self.forward_resblock(x, self.layer4)

        return skip

    def forward_down(self, image):
        skip = self.rn_backbone(image)
        skip = self.spp.forward(skip)
        return skip

    def forward(self, image):
        return self.forward_down(image)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

