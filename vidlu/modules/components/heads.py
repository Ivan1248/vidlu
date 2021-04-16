import math
from vidlu.utils.func import partial

import torch
from torch import nn
import torch.nn.functional as F

import vidlu.modules.elements as E
import vidlu.modules.components as vmc
from . import _default_factories as D


# Standard #########################################################################################

class ClassificationHead(E.Seq):
    def __init__(self, class_count):
        super().__init__(pre_logits_mean=nn.AdaptiveAvgPool2d((1, 1)),
                         logits=E.Linear(class_count))


class ClassificationHead1D(E.Seq):
    def __init__(self, class_count):
        super().__init__(logits=E.Linear(class_count))


class FixedSizeSegmentationHead(E.Seq):
    def __init__(self, class_count, shape=None, kernel_size=1, pre_activation=False,
                 norm_f=D.norm_f, act_f=E.ReLU):
        pre_act = dict(act=act_f(), norm=norm_f()) if pre_activation else {}
        super().__init__(
            **pre_act,
            logits=E.Conv(class_count, kernel_size=kernel_size, padding='half'),
            upsample=E.Func(
                partial(F.interpolate, size=shape, mode='bilinear', align_corners=False)))


class SegmentationHead(E.Module):
    def __init__(self, class_count, shape=None, kernel_size=1,
                 interpolate=partial(F.interpolate, mode='bilinear', align_corners=False)):
        super().__init__()
        self.logits = E.Conv(class_count, kernel_size=kernel_size, padding='half', bias=True)
        self.interpolate = interpolate
        self.shape = shape

    def forward(self, x, shape=None):
        return self.interpolate(self.logits(x), shape or self.shape)


class ResizingHead(E.Module):
    def __init__(self, shape=None, kernel_size=1,
                 interpolate=partial(F.interpolate, mode='bilinear', align_corners=False)):
        super().__init__()
        self.interpolate = interpolate
        self.shape = shape

    def forward(self, logits, shape=None):
        return self.interpolate(logits, shape or self.shape)


class RegressionHead(E.Seq):
    def __init__(self, class_count, shape):
        super().__init__(logits=E.Conv(class_count, kernel_size=1),
                         upsample=E.Func(partial(F.interpolate, size=shape, mode='bilinear',
                                                 align_corners=False)),
                         log_probs=nn.LogSoftmax(dim=1))


# Non-standard #####################################################################################

class TCSegmentationHead(E.Seq):  # TODO: deal with it?
    def __init__(self, class_count, shape, norm_f=D.norm_f, act_f=E.ReLU, convt_f=D.convt_f):
        super().__init__()
        self.shape = shape
        self.class_count = class_count
        self.norm_f, self.act_f, self.convt_f = norm_f, act_f, convt_f
        self.add(logits=E.Conv(class_count, kernel_size=1))

    def build(self, x):
        for i in range(round(math.log(self.shape[0] / (x.shape[2] * 2), 2))):
            self.add({f'norm{i}': self.norm_f(),
                      f'act{i}': self.act_f(),
                      f'conv{i}': self.convt_f(kernel_size=3,
                                               out_channels=x.shape[1] / 2 ** i,
                                               stride=2,
                                               padding=1)})
        self.add(upsample=E.Func(partial(F.interpolate, size=self.shape, mode='bilinear',
                                         align_corners=False)))


class ChannelMeanLogitsSplitter(E.Seq):
    """Extracts logits as channel means of the first class_count channels.

    Args:
        class_count (int): Number of classes.

    Returns:
        (logits, other1, other2) (tuple): cat([logits + other1, other2], dim=1)
        is the input, and the other logits has shape N×class_count×1×....
    """

    def __init__(self, class_count):
        super().__init__(spl=E.Split([class_count, ...]),
                         par=E.Parallel(mean_split=vmc.ChannelMeanSplitter(), id=E.Identity()),
                         restruct=E.Restruct("(logits, zl), zr", "logits, zl, zr"))


class ChannelMeanLogitsSplitter2(E.Seq):
    def __init__(self, class_count):
        super().__init__(spl=E.Split([class_count, ...]),
                         par=E.Parallel(
                             seq=E.Seq(
                                 mean_split=vmc.ChannelMeanSplitter(),
                                 par=E.Parallel(norm=E.BatchNorm(), id=E.Identity())),
                             id=E.Identity()),
                         restruct=E.Restruct("(logits, zl), zr", "logits, zl, zr"))


class AdvChannelMeanLogitsSplitter(E.Seq):
    """Extracts logits as channel means of the first class_count channels.

    Args:
        class_count (int): Number of classes.

    Returns:
        (logits, other1, other2) (tuple): cat([logits + other1, other2], dim=1)
        is the input, and the other logits has shape N×class_count×1×....
    """

    def __init__(self, class_count):
        super().__init__(spl=E.Split([class_count, ...]),
                         par=E.Parallel(mean_split=vmc.ChannelMeanSplitter(), id=E.Identity()),
                         restruct=E.Restruct("(logits, zl), zr", "logits, zl, zr"))
