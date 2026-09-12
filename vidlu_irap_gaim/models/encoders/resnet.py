from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ..resnet_backbone import resnet18
from .base import FrameEncoder, to_pixel_stats


class ResNetEncoder(FrameEncoder):
    """ResNet encoder with Spatial Pyramid Pooling (SPP) head for image sequences.

    Args:
        pixel_stats: Pixel normalization statistics `(mean, std)` to apply to input frames.
        pretrained: Whether to load ImageNet pre-trained weights in the backbone builder.
        builder: ResNet backbone constructor function.
        **builder_kwargs: Additional keyword arguments forwarded to `builder`.
    """

    def __init__(
        self,
        *,
        pixel_stats,
        pretrained: bool = True,
        builder: Callable[..., nn.Module] = resnet18,
        **builder_kwargs: Any,
    ):
        super().__init__()
        default_kwargs = dict(num_features=128, spp_grids=(6, 3, 2, 1), spp_square_grid=True, use_bn=True)
        default_kwargs.update(builder_kwargs)
        self.resnet = builder(pretrained=pretrained, **default_kwargs)
        self._set_pixel_stats(to_pixel_stats(pixel_stats))

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.resnet.rn_backbone(frames)
        pooled = self.resnet.spp(feature_map)
        return feature_map, pooled

    def pool_parameters(self) -> list[nn.Parameter]:
        return list(self.resnet.spp.parameters())
