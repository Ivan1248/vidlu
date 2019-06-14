import math
from functools import partial

from torch import nn
from torch.nn import functional as F

from vidlu.modules import Sequential, Linear, Conv, Func

from . import _default_factories as D


class ClassificationHead(Sequential):
    def __init__(self, class_count):
        super().__init__(pre_logits_mean=nn.AdaptiveAvgPool2d((1, 1)),
                         logits=Linear(class_count))


class SegmentationHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, mode='bilinear',
                                               align_corners=False)))


class TCSegmentationHead(Sequential):
    def __init__(self, class_count, shape, norm_f=D.norm_f, act_f=nn.ReLU, convt_f=D.convt_f):
        super().__init__()
        self.shape = shape
        self.class_count = class_count
        self.norm_f, self.act_f, self.convt_f = norm_f, act_f, convt_f

        self.add_module(logits=Conv(class_count, kernel_size=1))
        self.add_module(logits=Conv(class_count, kernel_size=1))

    def build(self, x):
        for i in range(round(math.log(self.shape[0] / (x.shape[2] * 2), 2))):
            self.add_module({f'norm{i}': self.norm_f(),
                             f'act{i}': self.act_f(),
                             f'conv{i}': self.convt_f(kernel_size=3,
                                                      out_channels=x.shape[1] / 2 ** i,
                                                      stride=2,
                                                      padding=1)})
        self.add_module(upsample=Func(partial(F.interpolate, size=self.shape, mode='bilinear',
                                              align_corners=False)))


class RegressionHead(Sequential):
    def __init__(self, class_count, shape):
        super().__init__(logits=Conv(class_count, kernel_size=1),
                         upsample=Func(partial(F.interpolate, size=shape, mode='bilinear',
                                               align_corners=False)),
                         log_probs=nn.LogSoftmax(dim=1))