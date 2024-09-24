from functools import partial
import typing as T
from pathlib import Path

import urllib
import torch

import vidlu.modules.components as vmc
from .models import SegmentationModel


class DeepLabV2RN101(SegmentationModel):
    def __init__(self, num_classes):
        super().__init__(
            backbone_f=partial(torch.hub.load, 'Ivan1248/cutmix-semisup-seg:master',
                               'resnet101_deeplab_imagenet', num_classes=num_classes),
            init=None,
            head_f=vmc.ResizingHead)


class DeepLabV3Plus(SegmentationModel):
    def __init__(self, backbone: T.Literal['resnet50', 'resnet101', 'xception', 'drn', 'mobilenet'],
                 output_stride: T.Literal[8, 16], num_classes: int, pretrained=False):
        super().__init__(
            backbone_f=partial(torch.hub.load, 'Ivan1248/Context-Aware-Consistency:master',
                               'DeepLabV3Plus', backbone=backbone, output_stride=output_stride,
                               num_classes=num_classes,
                               pretrained=pretrained),
            init=None,
            head_f=partial(vmc.SegmentationHead, class_count=num_classes))
