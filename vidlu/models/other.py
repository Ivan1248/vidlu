from functools import partial
import typing as T

import torch

import vidlu.modules.components as vmc
from .models import SegmentationModel


class DeepLabV2RN101(SegmentationModel):
    def __init__(self, num_classes):
        super().__init__(
            backbone_f=partial(torch.hub.load, 'Ivan1248/cutmix-semisup-seg',
                               'resnet101_deeplab_imagenet', num_classes=num_classes),
            init=None,
            head_f=vmc.ResizingHead)


class DeepLabV3Plus(SegmentationModel):
    def __init__(self, backbone: T.Literal['resnet50', 'resnet101', 'xception', 'drn', 'mobilenet'],
                 num_classes: int, pretrained=False):
        super().__init__(
            backbone_f=partial(torch.hub.load, 'Ivan1248/Context-Aware-Consistency',
                               'DeepLabV3Plus', backbone=backbone, num_classes=num_classes,
                               pretrained=pretrained),
            init=None,
            head_f=partial(vmc.SegmentationHead, class_count=num_classes))
