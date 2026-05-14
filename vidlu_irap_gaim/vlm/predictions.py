"""
Tensor and serialization utilities for VLM attribute predictions.

Converts parsed ``AttributePrediction`` dicts into metric-compatible
tensor tuples or JSON-serializable dicts.
"""

from typing import Sequence

import torch

from .response_parser import AttributePrediction


def predictions_to_output_tuple(
    predictions: dict[str, AttributePrediction],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_order: Sequence[str],
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    *,
    allow_invalid: bool = False,
    required_attrs: set[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Convert parsed predictions to metric-compatible output format.

    The metrics system expects:
        iter_result.out = tuple of (B, K_i) tensors
    where out[i].argmax(1) gives the predicted class for attribute i.

    We create one-hot tensors so argmax returns exactly pred_idx.

    Args:
        predictions: Dictionary of AttributePrediction objects.
        attr_to_value_to_class_idx: Mapping for class counts.
        attrs_order: Order of attributes (must match dataset order).
        batch_size: Batch size (typically 1 for VLM inference).
        device: Target device for tensors.

    Returns:
        Tuple of (B, K_i) tensors, one per attribute.
    """
    out_tensors = []

    for attr_name in attrs_order:
        num_classes = len(attr_to_value_to_class_idx.get(attr_name, {}))
        if num_classes == 0:
            # Attribute not in metadata, create dummy
            num_classes = 1

        # Create logits tensor (one-hot style)
        logits = torch.zeros(batch_size, num_classes, device=device)

        pred = predictions.get(attr_name)
        is_required = True if required_attrs is None else (attr_name in required_attrs)

        if pred is None:
            if is_required and not allow_invalid:
                raise ValueError(
                    f"Missing prediction for required attribute {attr_name!r}. "
                    f"Set allow_invalid=True to keep running (will bias metrics), "
                    f"or ensure the VLM prompt covers this attribute."
                )
            # Non-required attribute (or explicitly allowed): fill with class 0.
            logits[:, 0] = 1.0
        elif 0 <= pred.pred_idx < num_classes:
            logits[:, pred.pred_idx] = 1.0
        else:
            if is_required and not allow_invalid:
                raise ValueError(
                    f"Invalid prediction for required attribute {attr_name!r}: "
                    f"pred={(pred.pred_value, pred.pred_idx)}; num_classes={num_classes}. "
                    f"Set allow_invalid=True to keep running (will bias metrics)."
                )
            logits[:, 0] = 1.0

        out_tensors.append(logits)

    return tuple(out_tensors)


def convert_attribute_predictions_to_standard_format(
    batch_predictions: list[dict[str, AttributePrediction]],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_order: Sequence[str],
    device: str | torch.device = "cpu",
    *,
    allow_invalid: bool = True,
    required_attrs: set[str] | None = None,
    attrs_to_include: set[str] | Sequence[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Convert a batch of parsed predictions to metric-compatible output format.

    The metrics system expects:
        iter_result.out = tuple of (B, K_i) tensors
    where out[i].argmax(1) gives the predicted class for attribute i.

    Args:
        batch_predictions: List of prediction dicts, one per sample.
        attr_to_value_to_class_idx: Mapping for class counts.
        attrs_order: Full attribute order (must match dataset order).
        device: Target device for tensors.
        allow_invalid: If True, invalid/missing predictions fall back to class 0.
        required_attrs: Attributes that must have valid predictions (when allow_invalid=False).
        attrs_to_include: Attributes with real predictions. Others get dummy (B, 1) tensors.
            If None, uses all attributes from attrs_order.

    Returns:
        Tuple of (B, K_i) tensors, one per attribute.
    """
    batch_size = len(batch_predictions)
    attrs_to_include_set = set(attrs_to_include) if attrs_to_include else set(attrs_order)

    out_tensors = []
    for attr_name in attrs_order:
        num_classes = len(attr_to_value_to_class_idx.get(attr_name, {}))
        if num_classes == 0:
            num_classes = 1

        logits = torch.zeros(batch_size, num_classes, device=device)

        if attr_name in attrs_to_include_set:
            is_required = required_attrs is None or attr_name in required_attrs
            for sample_idx in range(batch_size):
                pred = batch_predictions[sample_idx].get(attr_name)
                if pred is None:
                    if is_required and not allow_invalid:
                        raise ValueError(
                            f"Missing prediction for required attribute {attr_name!r} at sample {sample_idx}."
                        )
                    logits[sample_idx, 0] = 1.0
                elif 0 <= pred.pred_idx < num_classes:
                    logits[sample_idx, pred.pred_idx] = 1.0
                else:
                    if is_required and not allow_invalid:
                        raise ValueError(
                            f"Invalid prediction for {attr_name!r} at sample {sample_idx}: "
                            f"pred_idx={pred.pred_idx}, num_classes={num_classes}."
                        )
                    logits[sample_idx, 0] = 1.0
        else:
            # Dummy for non-included attributes (not accessed by metrics)
            logits = torch.zeros(batch_size, 1, device=device)

        out_tensors.append(logits)

    return tuple(out_tensors)


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
