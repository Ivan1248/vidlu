from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import init

from .encoders import AttentionBlock


def build_classification_heads(
    input_dim: int,
    class_counts: tuple[int, ...],
) -> nn.ModuleList:
    """
    Build classification heads.

    Args:
        input_dim: Input dimension for the heads
        class_counts: Tuple of number of classes per attribute

    Returns:
        Heads ModuleList
    """
    heads = nn.ModuleList()
    for k in class_counts:
        fc = nn.Linear(input_dim, k)
        init.xavier_uniform_(fc.weight)
        init.zeros_(fc.bias)
        heads.append(fc)
    return heads


def build_attention_blocks(
    per_frame_attn_dim: int,
    class_counts: tuple[int, ...],
) -> nn.ModuleList:
    """
    Build attention blocks.

    Args:
        per_frame_attn_dim: Feature dimension per frame for attention
        class_counts: Tuple of number of classes per attribute

    Returns:
        Attention blocks ModuleList
    """
    if per_frame_attn_dim <= 0:
        raise ValueError("Attention enabled but encoder produced zero channels.")

    return nn.ModuleList([AttentionBlock(per_frame_attn_dim) for _ in class_counts])


class ImageSequenceClassifier(nn.Module):
    """
    ViDLU-compatible multi-attribute classifier over sequences using SPP features.

    Inputs:
        - During training: a tensor shaped (B, S, C, H, W)
        - In tests / direct calls: a dict with key 'rgb' shaped (B, S, C, H, W)

    Output: tuple of logits tensors, one per attribute, each shaped (B, K_i)
    """

    def __init__(
        self,
        class_counts: tuple[int, ...],
        sequence_length: int,
        *,
        attention: bool = False,
        encoder_f: Callable[..., nn.Module],
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.attention = attention
        if isinstance(encoder_f, nn.Module):
            self.frame_encoder = encoder_f
        elif callable(encoder_f):
            self.frame_encoder = encoder_f()
        else:
            raise TypeError(f"encoder_f must be a FrameEncoder instance or a callable factory, got {type(encoder_f)}")
        self.class_counts = class_counts

        dummy_input = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            feature_map, pooled = self.frame_encoder(dummy_input)
        spp_output_dim = pooled.shape[1]
        per_frame_attn_dim = feature_map.shape[1] if attention else 0

        head_input_dim = (
            sequence_length * spp_output_dim
            if not attention
            else sequence_length * (spp_output_dim + per_frame_attn_dim)
        )
        self.heads = build_classification_heads(head_input_dim, self.class_counts)

        if attention:
            self.attn_blocks = build_attention_blocks(per_frame_attn_dim, self.class_counts)
        else:
            self.attn_blocks = None

    def forward(self, x, return_features: bool = False):
        """
        Accept either a tensor (training path) or a dict with 'rgb' (equivalence tests).
        """
        if isinstance(x, dict):
            if "rgb" not in x:
                raise KeyError(f"Expected key 'rgb' in input dict, got keys: {list(x.keys())}")
            rgb = x["rgb"]
        else:
            rgb = x

        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float()
        if not isinstance(rgb, torch.Tensor):
            raise TypeError(f"Expected rgb to be a Tensor or numpy array, got {type(rgb)}")

        B, S = rgb.shape[:2]
        frame_results = [self.frame_encoder(rgb[:, i]) for i in range(S)]
        feats_before = [feature_map for feature_map, _ in frame_results]
        feats = [pooled for _, pooled in frame_results]

        if len(feats) == 0:
            raise ValueError("Expected at least one frame per sequence.")

        if S != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {S}")

        seq_feat = torch.cat(feats, dim=1)

        if self.attention:
            outs = [
                fc(torch.cat([attn_block(f) for f in feats_before] + [seq_feat], dim=1))
                for fc, attn_block in zip(self.heads, self.attn_blocks)
            ]
        else:
            outs = [fc(seq_feat) for fc in self.heads]

        if return_features:
            return tuple(outs), seq_feat
        return tuple(outs)

    def get_trainable_parameters(self):
        """Return trainable parameters following original logic: heads + pooling + attention blocks"""
        trainable_params = []
        for head in self.heads:
            trainable_params.extend(head.parameters())
        trainable_params.extend(self.frame_encoder.pooling_parameters())
        if self.attn_blocks is not None:
            for attn_block in self.attn_blocks:
                trainable_params.extend(attn_block.parameters())
        return list(trainable_params)

