"""
Tensor and serialization utilities for VLM attribute predictions.

Converts parsed ``AttributePrediction`` dicts into metric-compatible
tensor tuples or JSON-serializable dicts.
"""

from typing import Sequence

import torch

from .response_parser import AttributePrediction


def is_usable_prediction(pred: AttributePrediction | None, num_classes: int) -> bool:
    """Whether a response can be scored: present, and naming a class of the attribute.

    The one definition of "usable" that the invalid rate, the two scorings and the
    per-record analysis all read, so that they cannot disagree on it.
    """
    return pred is not None and 0 <= pred.pred_idx < num_classes


def attribute_predictions_to_one_hot_outputs(
    batch_predictions: list[dict[str, AttributePrediction]],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_order: Sequence[str],
    device: str | torch.device = "cpu",
    *,
    attrs_to_include: set[str] | Sequence[str] | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Convert a batch of parsed predictions to metric-compatible output format.

    The metrics system expects:
        iter_result.out = tuple of (B, K_i) tensors
    where out[i].argmax(1) gives the predicted class for attribute i.

    We create one-hot tensors so argmax returns exactly the predicted index. An
    attribute with no usable prediction (omitted, or a value matching nothing in
    the value set) is one-hot at class 0 and flagged in the returned mask; how to
    score it is the caller's decision – see ``vlm.scoring.update_both_scorings``,
    which scores it both as class 0 and as excluded.

    Args:
        batch_predictions: List of prediction dicts, one per sample.
        attr_to_value_to_class_idx: Mapping for class counts.
        attrs_order: Full attribute order (must match dataset order).
        device: Target device for tensors.
        attrs_to_include: Attributes with real predictions. Others get dummy (B, 1) tensors.
            If None, uses all attributes from attrs_order.

    Returns:
        ``(out_tensors, is_invalid)``. *out_tensors* is a tuple of (B, K_i) tensors,
        one per attribute in ``attrs_order``. *is_invalid* is a (B, A) boolean tensor
        marking the (sample, attribute) pairs without a usable prediction; it is
        False for attributes outside ``attrs_to_include``.
    """
    batch_size = len(batch_predictions)
    attrs_to_include_set = set(attrs_to_include) if attrs_to_include else set(attrs_order)
    is_invalid = torch.zeros(batch_size, len(attrs_order), dtype=torch.bool, device=device)

    out_tensors = []
    for attr_idx, attr_name in enumerate(attrs_order):
        num_classes = max(1, len(attr_to_value_to_class_idx.get(attr_name, {})))

        if attr_name not in attrs_to_include_set:
            # Dummy for non-included attributes (not accessed by metrics)
            out_tensors.append(torch.zeros(batch_size, 1, device=device))
            continue

        logits = torch.zeros(batch_size, num_classes, device=device)
        for sample_idx, predictions in enumerate(batch_predictions):
            pred = predictions.get(attr_name)
            if is_usable_prediction(pred, num_classes):
                logits[sample_idx, pred.pred_idx] = 1.0
            else:
                logits[sample_idx, 0] = 1.0
                is_invalid[sample_idx, attr_idx] = True
        out_tensors.append(logits)

    return tuple(out_tensors), is_invalid


def predictions_to_json_serializable(
    predictions: dict[str, AttributePrediction],
) -> dict[str, dict[str, any]]:
    """Convert predictions to JSON-serializable format for saving."""
    return {
        attr_name: {
            "pred_value": pred.pred_value,
            "pred_idx": pred.pred_idx,
            "confidence": pred.confidence,
        }
        for attr_name, pred in predictions.items()
    }
