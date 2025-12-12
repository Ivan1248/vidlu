from __future__ import annotations

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softmax


class AttentionBlock(nn.Module):
    """Simple per-frame attention head used by ``ImageSequenceClassifier``."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        q = nn.Parameter(torch.randn(1, state_dim, 1))
        init.xavier_uniform_(q)
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape x: (B, C, H, W) -> (B, C, HW)
        x = x.view(tuple(x.shape[:2]) + (-1,))
        r = self.q * x
        unnormalized_attn_map = r.sum(1)
        attn_map = softmax(unnormalized_attn_map, dim=1)
        return (torch.unsqueeze(attn_map, 1) * x).sum(2)





