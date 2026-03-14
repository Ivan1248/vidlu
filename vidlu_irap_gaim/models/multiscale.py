"""Multi-scale inference for ImageSequenceClassifier.

Provides wrappers that handle sequence input (B, S, C, H, W) and multi-attribute
output, applying the model at multiple scales and averaging probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _multi_attribute_softmax(logits_tuple: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Apply softmax to each attribute's logits in a tuple output."""
    return tuple(F.softmax(logits, dim=1) for logits in logits_tuple)


class MultiScaleSequenceInference(nn.Module):
    """Multi-scale wrapper for ImageSequenceClassifier.

    Handles sequence input (B, S, C, H, W) and multi-attribute output.
    Applies model at multiple scales and averages probabilities.

    Args:
        base_model: The ImageSequenceClassifier instance.
        scales: Tuple of scale factors. Default matches SemisupMultiScaleTeacherStep.
    """

    def __init__(
        self,
        base_model: nn.Module,
        scales: tuple[float, ...] = (1.0, 0.75, 1 / 0.75),
    ):
        super().__init__()
        self.base_model = base_model
        self.scales = scales

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass with multi-scale averaging.

        Args:
            x: Input tensor of shape (B, S, C, H, W).

        Returns:
            Tuple of averaged probability tensors, one per attribute.
        """
        all_probs: list[tuple[torch.Tensor, ...]] = []

        for scale in self.scales:
            if scale != 1.0:
                x_scaled = self._scale_sequence(x, scale)
            else:
                x_scaled = x

            logits = self.base_model(x_scaled)
            probs = _multi_attribute_softmax(logits)
            all_probs.append(probs)

        # Average probabilities across scales for each attribute
        num_attributes = len(all_probs[0])
        return tuple(torch.stack([probs[i] for probs in all_probs]).mean(dim=0) for i in range(num_attributes))

    @staticmethod
    def _scale_sequence(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale all frames in a sequence.

        Args:
            x: Input tensor of shape (B, S, C, H, W).
            scale: Scale factor.

        Returns:
            Scaled tensor.
        """
        B, S, C, H, W = x.shape
        # Reshape to (B*S, C, H, W) for interpolation
        x_flat = x.view(B * S, C, H, W)
        x_scaled = F.interpolate(
            x_flat,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
        # Reshape back to (B, S, C, H', W')
        _, _, H_new, W_new = x_scaled.shape
        return x_scaled.view(B, S, C, H_new, W_new)
