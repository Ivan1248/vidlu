from __future__ import annotations

from typing import Sequence
import torch
import torch.nn.functional as F


class MultiAttributeCrossEntropyLoss:
    """
    Callable loss wrapper that mirrors multi_attribute_cross_entropy while allowing
    attribute filtering and per-attribute class weight injection without swapping loss functions.
    """

    supports_dynamic_class_weights = True

    def __init__(
        self,
        reduction: str = "mean",
        attrs_idx: Sequence[int] | None = None,
    ) -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.default_reduction = reduction
        self.attrs_idx = list(attrs_idx) if attrs_idx is not None else None
        self._attr_idx_to_class_weights: dict[int, torch.Tensor] | None = None

    def set_attrs_idx(self, attrs_idx: Sequence[int]) -> None:
        """Persist attribute filtering so extensions can control which attributes participate."""
        self.attrs_idx = list(attrs_idx)

    def set_class_weights(self, attr_idx_to_class_weights: dict[int, torch.Tensor] | None) -> None:
        """Store per-attribute weight tensors (dict mapping GLOBAL attr_idx to weights)."""
        if attr_idx_to_class_weights is None:
            self._attr_idx_to_class_weights = None
        else:
            self._attr_idx_to_class_weights = {
                k: w.clone().detach() for k, w in attr_idx_to_class_weights.items()
            }

    def _weight_for(self, global_attr_idx: int, device: torch.device) -> torch.Tensor | None:
        if self._attr_idx_to_class_weights is None:
            return None
        weight = self._attr_idx_to_class_weights.get(global_attr_idx)
        if weight is None:
            return None
        return weight.to(device)

    def __call__(
        self,
        outputs: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        reduction: str | None = None,
        attrs_idx: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("outputs should be a tuple/list of logits per attribute")

        effective_reduction = reduction or self.default_reduction
        if effective_reduction not in ("mean", "sum"):
            raise ValueError(f"Unsupported reduction: {effective_reduction}")

        attr_indices: Sequence[int]
        if attrs_idx is not None:
            attr_indices = attrs_idx
        elif self.attrs_idx is not None:
            attr_indices = self.attrs_idx
        else:
            attr_indices = list(range(len(outputs)))

        per_attr_losses = []
        for global_attr_idx in attr_indices:
            weight = self._weight_for(global_attr_idx, outputs[global_attr_idx].device)
            per_attr_losses.append(
                F.cross_entropy(outputs[global_attr_idx], targets[:, global_attr_idx], reduction=effective_reduction, weight=weight)
            )

        if len(per_attr_losses) == 0:
            raise ValueError("No attributes selected for loss computation")

        if effective_reduction == "mean":
            return torch.stack(per_attr_losses).mean()
        return torch.stack(per_attr_losses).sum()


def multi_attribute_cross_entropy(
    outputs: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    reduction: str = "mean",
    attrs_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Cross-entropy over a tuple of attribute logits and a (B, A) target tensor where A=len(outputs).
    Supports reduction argument to be consistent with ViDLU training steps.
    - reduction="mean": mean over batch per attribute, then mean over attributes (default)
    - reduction="sum": sum over batch per attribute, then sum over attributes
    
    Args:
        outputs: Tuple of logits tensors, one per attribute.
        targets: Target tensor of shape (B, A) where A=len(outputs).
        reduction: "mean" or "sum".
        attrs_idx: Optional list of attribute indices to include. If None, includes all attributes.
    """
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("outputs should be a tuple/list of logits per attribute")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    
    if attrs_idx is None:
        attrs_idx = list(range(len(outputs)))
    
    per_attr_losses = []
    for i in attrs_idx:
        per_attr_losses.append(F.cross_entropy(outputs[i], targets[:, i], reduction=reduction))
    
    if len(per_attr_losses) == 0:
        raise ValueError("No attributes selected for loss computation")
    
    if reduction == "mean":
        return torch.stack(per_attr_losses).mean()
    else:
        return torch.stack(per_attr_losses).sum()


