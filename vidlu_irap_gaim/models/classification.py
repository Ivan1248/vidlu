from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import init

import vidlu.modules.elements as E

from .encoders import AttentionBlock, FrameEncoder, TrainabilityMode


def build_classification_heads(
    input_dim: int,
    class_counts: tuple[int, ...],
) -> nn.ModuleList:
    """
    Builds classification heads.

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
    Builds attention blocks.

    Args:
        per_frame_attn_dim: Feature dimension per frame for attention
        class_counts: Tuple of number of classes per attribute

    Returns:
        Attention blocks ModuleList
    """
    if per_frame_attn_dim <= 0:
        raise ValueError("Attention enabled but encoder produced zero channels.")

    return nn.ModuleList([AttentionBlock(per_frame_attn_dim) for _ in class_counts])


class ImageSequenceClassifier(E.Module):
    """Multi-attribute classifier for image sequences using a FrameEncoder backbone.

    Encodes input image frames with `frame_encoder`, concatenates pooled temporal features,
    and applies per-attribute classification heads (with optional spatial attention).

    The heads are built on the first call (see `build`), sized from the encoder's actual
    output on that input. `vidlu.factories.get_model` performs such a call with a real
    batch; anywhere else the model has to be called once before its heads are used (e.g.
    before creating an optimizer or loading a state dict).

    Args:
        class_counts: Tuple containing the number of classes for each attribute.
        sequence_length: Number of frames in each input sequence.
        attention: If True, applies per-attribute spatial attention on encoder feature maps.
        encoder_f: Factory producing a `FrameEncoder`.
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
        self.frame_encoder = encoder_f()
        if not isinstance(self.frame_encoder, FrameEncoder):
            raise TypeError(
                f"encoder_f must produce a FrameEncoder, got {type(self.frame_encoder).__name__}.")
        self.class_counts = class_counts
        self.heads: nn.ModuleList | None = None
        self.attn_blocks: nn.ModuleList | None = None

    def _frames_from_input(self, x) -> torch.Tensor:
        """Extracts the `(B, S, C, H, W)` frame tensor from a tensor, array, or 'rgb' dict."""
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

        if rgb.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {rgb.shape[1]}")
        return rgb

    def build(self, x):
        """Sizes the heads from the encoder's output widths on the first input.

        The widths of every supported encoder are independent of the input resolution,
        so the shapes of this input carry over to all later ones.
        """
        rgb = self._frames_from_input(x)
        # Only shapes are needed: in eval mode the pass leaves BatchNorm running statistics
        # (and any other train-mode state) untouched.
        was_training = self.frame_encoder.training
        self.frame_encoder.eval()
        try:
            with torch.no_grad():
                feature_map, pooled = self.frame_encoder(rgb.flatten(0, 1))
        finally:
            self.frame_encoder.train(was_training)
        pooled_dim = pooled.shape[1]
        per_frame_attn_dim = feature_map.shape[1] if self.attention else 0
        head_input_dim = self.sequence_length * (pooled_dim + per_frame_attn_dim)
        self.heads = build_classification_heads(head_input_dim, self.class_counts)
        if self.attention:
            self.attn_blocks = build_attention_blocks(per_frame_attn_dim, self.class_counts)

    def forward(self, x, return_features: bool = False):
        """Forward pass across the sequence.

        Args:
            x: Input tensor of shape `(batch_size, sequence_length, C, H, W)`, or a dict
                with an 'rgb' entry of that shape.
            return_features: If True, returns a tuple `(logits_tuple, pooled_features)`.

        Returns:
            Tuple of per-attribute logit tensors, each of shape `(batch_size, num_classes)`.
            If `return_features` is True, returns `(logits_tuple, seq_features)`.
        """
        rgb = self._frames_from_input(x)
        B, S = rgb.shape[:2]

        # The whole sequence is encoded in one call, with frames folded into the batch
        # dimension, rather than one call per frame. Same arithmetic, but a transformer
        # backbone then sees B*S images per forward instead of B, which is what makes a
        # useful batch size reachable for the larger encoders.
        feature_maps, pooled = self.frame_encoder(rgb.flatten(0, 1))
        seq_feat = pooled.reshape(B, -1)  # [frame 0 | frame 1 | ...] per example

        if self.attention:
            per_frame_maps = feature_maps.reshape(B, S, *feature_maps.shape[1:])
            outs = [
                fc(torch.cat([attn_block(per_frame_maps[:, i]) for i in range(S)] + [seq_feat],
                             dim=1))
                for fc, attn_block in zip(self.heads, self.attn_blocks)
            ]
        else:
            outs = [fc(seq_feat) for fc in self.heads]

        if return_features:
            return tuple(outs), seq_feat
        return tuple(outs)

    def head_modules(self) -> list[nn.Module]:
        """Returns the classification head and attention modules."""
        if not self.is_built():
            raise RuntimeError(
                f"{type(self).__name__} has no heads yet: they are built on the first call."
                f" Call the model on a batch before using its heads.")
        return [m for m in (self.heads, self.attn_blocks) if m is not None]

    def head_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.head_modules() for p in m.parameters()]

    def set_encoder_trainable(self, mode: TrainabilityMode) -> None:
        """Sets trainability mode for the frame encoder backbone while keeping heads trainable.

        Args:
            mode: Backbone trainability mode ('none', 'pool', 'lora', 'last_blocks', 'all').
        """
        self.frame_encoder.set_trainable(mode)
        for m in self.head_modules():
            m.requires_grad_(True)

    def load_state_dict(self, state_dict, strict: bool = True):
        # `E.Module` would otherwise defer the load to the first call and drop `strict`
        # and the missing/unexpected-keys result. Requiring an explicit build keeps
        # partial (`strict=False`) loads honest.
        if not self.is_built():
            raise RuntimeError(
                f"{type(self).__name__} is not built yet, so a state dict cannot be loaded into"
                f" it. Call the model on a batch first.")
        return super().load_state_dict(state_dict, strict=strict)
