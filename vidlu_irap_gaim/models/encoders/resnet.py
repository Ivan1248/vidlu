from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ...resnet_backbone import resnet18


class ResNetEncoder(nn.Module):
    """IRAP GAIM ResNet encoder with Spatial Pyramid Pooling head."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        builder: Callable[..., nn.Module] = resnet18,
        **builder_kwargs: Any,
    ):
        super().__init__()
        default_kwargs = dict(num_features=128, spp_grids=(6, 3, 2, 1), spp_square_grid=True, use_bn=True)
        default_kwargs.update(builder_kwargs)
        self.resnet = builder(pretrained=pretrained, **default_kwargs)

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.resnet.rn_backbone(frame)
        pooled = self.resnet.spp(feature_map)
        return feature_map, pooled

    def pooling_parameters(self) -> list[nn.Parameter]:
        return list(self.resnet.spp.parameters())





