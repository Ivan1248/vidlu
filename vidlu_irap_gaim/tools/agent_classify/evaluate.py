"""
Evaluate predictions produced by a workspace agent (e.g. Google Antigravity) against
iRAP ground-truth labels.

The agent is expected to write a JSON file mapping segment IDs to attribute value strings::

    {
        "seg_id_1": {"Attribute Name": "Value A", ...},
        "seg_id_2": {"Attribute Name": "Value B", ...}
    }

Driven by the ``agent_classify evaluate`` frontend (see ``__main__.py``); the public
entry point here is :func:`evaluate_agent_predictions`. It loads the ground-truth
dataset, parses the predicted values using the same fuzzy-matching parser as
``vlm_inference.py``, computes metrics with ``get_irap_metrics``, and saves
``predictions.json`` + ``summary.json`` in the same format as
``vidlu_irap_gaim.tools.vlm_inference.run_evaluation``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AgentEvaluationResult:
    """Summary of agent-prediction evaluation."""

    num_samples_in_split: int
    num_samples_with_predictions: int
    num_samples_completed: int
    num_valid_predictions: int
    metrics: dict[str, float] | None
    output_dir: Path
    predictions_file: Path


def _parse_agent_prediction(
    pred_dict: dict[str, Any],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_to_include: list[str],
) -> dict:
    """Parse a single segment's agent prediction dict into AttributePrediction objects.

    Reuses ``parse_vlm_response`` by serialising the dict back to JSON text — this
    keeps the fuzzy-matching and key-remapping logic in one place.
    """
    from vidlu_irap_gaim.vlm.response_parser import parse_vlm_response

    raw_text = json.dumps(pred_dict, ensure_ascii=False)
    return parse_vlm_response(
        raw_text,
        attr_to_value_to_class_idx,
        attrs_to_include=attrs_to_include,
        output_format="json",
    )


def evaluate_agent_predictions(
    predictions_file: str | Path,
    dataset_name: str = "bih",
    split: str = "test",
    output_dir: str | Path = "agent_eval_results",
    allow_invalid_predictions: bool = True,
) -> AgentEvaluationResult:
    """Evaluate agent predictions against ground-truth labels.

    Args:
        predictions_file: Path to the agent's predictions JSON (the file
            ``prepare_agent_tasks`` told the agent to write).
        dataset_name: ``"bih"`` or ``"vietnam"``.
        split: Dataset split to evaluate against.
        output_dir: Where to write ``predictions.json`` and ``summary.json``.
        allow_invalid_predictions: If True, unparseable values fall back to class 0
            (biases metrics but keeps evaluation running). Default True for agent output.

    Returns:
        AgentEvaluationResult with evaluation summary.
    """
    from vidlu_irap_gaim.vlm.predictions import (
        predictions_to_output_tuple,
        predictions_to_json_serializable,
    )
    from ._common import load_split_and_attrs
    from vidlu_irap_gaim.metrics import get_irap_metrics
    from vidlu.utils.collections import NameDict

    predictions_file = Path(predictions_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load agent predictions
    print(f"Loading agent predictions from {predictions_file}...")
    with open(predictions_file, "r", encoding="utf-8") as f:
        raw_predictions: dict[str, Any] = json.load(f)
    print(f"  {len(raw_predictions)} segments in predictions file.")

    print(f"Loading {dataset_name.upper()} dataset (split={split})...")
    dataset, attr_to_value_to_class_idx, attrs_to_include = load_split_and_attrs(
        dataset_name, split
    )
    print(f"  {len(dataset)} segments in split.")
    attrs_order = list(attr_to_value_to_class_idx.keys())

    # Set up metrics
    metrics = get_irap_metrics(dataset, attrs_to_include=tuple(attrs_to_include))
    for m in metrics:
        m.reset()

    # Evaluate
    all_output_predictions: dict[str, Any] = {}
    num_with_predictions = 0
    num_valid = 0
    num_completed = 0

    print(f"Evaluating {len(dataset)} segments...")
    import torch
    from tqdm import tqdm

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        segment_id = sample.get("segment_id", str(idx)) if hasattr(sample, "get") else str(idx)
        target = sample.get("target") if hasattr(sample, "get") else None

        agent_pred = raw_predictions.get(segment_id)

        if agent_pred is None:
            # No prediction for this segment — treat as all-invalid
            all_output_predictions[segment_id] = {"error": "no_prediction"}
            num_completed += 1
            # Still update metrics with dummy predictions so denominator is correct
            dummy_preds = {}
            out = predictions_to_output_tuple(
                dummy_preds,
                attr_to_value_to_class_idx,
                attrs_order,
                batch_size=1,
                device="cpu",
                allow_invalid=True,
            )
            if target is not None:
                t = target.unsqueeze(0) if isinstance(target, torch.Tensor) else target
                for m in metrics:
                    m.update(NameDict(out=out, target=t))
            continue

        num_with_predictions += 1

        # Handle "image_not_found" sentinel
        if agent_pred.get("image_not_found"):
            all_output_predictions[segment_id] = {"error": "image_not_found"}
            num_completed += 1
            dummy_preds = {}
            out = predictions_to_output_tuple(
                dummy_preds,
                attr_to_value_to_class_idx,
                attrs_order,
                batch_size=1,
                device="cpu",
                allow_invalid=True,
            )
            if target is not None:
                t = target.unsqueeze(0) if isinstance(target, torch.Tensor) else target
                for m in metrics:
                    m.update(NameDict(out=out, target=t))
            continue

        # Parse predicted values
        predictions = _parse_agent_prediction(
            agent_pred, attr_to_value_to_class_idx, attrs_to_include
        )

        valid_count = sum(1 for p in predictions.values() if p.pred_idx >= 0)
        if valid_count > 0:
            num_valid += 1

        # Log invalid predictions
        invalid = {a: p for a, p in predictions.items() if p.pred_idx < 0}
        if invalid:
            print(
                f"  [INVALID] {segment_id}: {len(invalid)}/{len(predictions)} attrs unparsed"
                f" — e.g. {next(iter(invalid.values())).pred_value!r}"
            )

        all_output_predictions[segment_id] = {
            "predictions": predictions_to_json_serializable(predictions),
        }

        # Update metrics
        if target is not None:
            out = predictions_to_output_tuple(
                predictions,
                attr_to_value_to_class_idx,
                attrs_order,
                batch_size=1,
                device="cpu",
                allow_invalid=allow_invalid_predictions,
                required_attrs=set(attrs_to_include),
            )
            t = target.unsqueeze(0) if isinstance(target, torch.Tensor) else target
            for m in metrics:
                m.update(NameDict(out=out, target=t))

        num_completed += 1

    # Compute metrics
    computed_metrics: dict[str, Any] = {}
    for m in metrics:
        computed_metrics.update(m.compute())

    print("\n=== Evaluation Metrics ===")
    coverage = num_with_predictions / len(dataset) if len(dataset) > 0 else 0.0
    print(f"  Coverage: {num_with_predictions}/{len(dataset)} segments ({coverage:.1%})")
    for k, v in computed_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            avg = sum(v.values()) / len(v) if v else 0.0
            print(f"  {k} (avg): {avg:.4f}")

    # Save outputs
    predictions_out_file = output_dir / "predictions.json"
    with open(predictions_out_file, "w", encoding="utf-8") as f:
        json.dump(all_output_predictions, f, indent=2, ensure_ascii=False)

    summary = {
        "num_samples_in_split": len(dataset),
        "num_samples_with_predictions": num_with_predictions,
        "num_samples_completed": num_completed,
        "num_valid_predictions": num_valid,
        "coverage": coverage,
        "split": split,
        "dataset": dataset_name,
        "predictions_source": str(predictions_file),
        "metrics": {
            k: (v if isinstance(v, (int, float, type(None))) else dict(v))
            for k, v in computed_metrics.items()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  predictions.json: {len(all_output_predictions)} entries")
    print("  summary.json")

    return AgentEvaluationResult(
        num_samples_in_split=len(dataset),
        num_samples_with_predictions=num_with_predictions,
        num_samples_completed=num_completed,
        num_valid_predictions=num_valid,
        metrics=computed_metrics,
        output_dir=output_dir,
        predictions_file=predictions_out_file,
    )
