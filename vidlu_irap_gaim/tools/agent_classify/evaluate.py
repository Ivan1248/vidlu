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
``vidlu_irap_gaim.tools.vlm_inference.run_evaluation`` -- including both scorings of an
unusable response and the invalid rate, so agent runs and VLM runs are comparable.
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
    num_scored_attribute_responses: int
    num_invalid_attribute_responses: int
    metrics: dict[str, float] | None
    metrics_excluding_invalid: dict[str, float] | None
    output_dir: Path
    predictions_file: Path

    @property
    def invalid_rate(self) -> float:
        """Fraction of (segment, attribute) responses that were unusable.

        Derived rather than stored, matching `vlm_inference.EvaluationResult`, so
        it cannot be constructed inconsistently with the two counts.
        """
        if self.num_scored_attribute_responses == 0:
            return float("nan")
        return self.num_invalid_attribute_responses / self.num_scored_attribute_responses


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
) -> AgentEvaluationResult:
    """Evaluate agent predictions against ground-truth labels.

    An attribute the agent gave no usable response for is scored both ways, because
    neither scoring subsumes the other (see ``vlm.scoring``):
    as class 0, which is what the training-time eval does, and excluded, which measures
    skill conditional on a format-compliant response. Both are reported, beside the invalid rate
    the second one has to be read against.

    Args:
        predictions_file: Path to the agent's predictions JSON (the file
            ``prepare_agent_tasks`` told the agent to write).
        dataset_name: ``"bih"`` or ``"vietnam"``.
        split: Dataset split to evaluate against.
        output_dir: Where to write ``predictions.json`` and ``summary.json``.

    Returns:
        AgentEvaluationResult with evaluation summary.
    """
    from vidlu_irap_gaim.vlm.predictions import predictions_to_json_serializable
    from vidlu_irap_gaim.vlm.scoring import (count_scored_and_invalid_responses,
                                             metrics_to_json_dict, print_metrics,
                                             update_both_scorings)
    from ._common import load_split_and_attrs
    from vidlu_irap_gaim.metrics import get_irap_metrics

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

    # Two metric sets, one per scoring of an unusable response. Both are reported. Parsed text
    # gives one-hot outputs, so no probabilistic metrics.
    metrics = get_irap_metrics(dataset, attrs_to_include=tuple(attrs_to_include),
                               output_kind="hard")
    metrics_excluding_invalid = get_irap_metrics(
        dataset, attrs_to_include=tuple(attrs_to_include), output_kind="hard")
    for m in (metrics, metrics_excluding_invalid):
        m.reset()

    # Evaluate
    all_output_predictions: dict[str, Any] = {}
    num_with_predictions = 0
    num_valid = 0
    num_completed = 0
    num_scored_responses = 0
    num_invalid_responses = 0

    print(f"Evaluating {len(dataset)} segments...")
    import torch
    from tqdm import tqdm

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        segment_id = sample.get("segment_id", str(idx)) if hasattr(sample, "get") else str(idx)
        target = sample.get("target") if hasattr(sample, "get") else None

        agent_pred = raw_predictions.get(segment_id)

        # A segment the agent never responded to -- it produced nothing, or it reported the
        # image missing -- is an all-invalid sample, represented by an empty prediction
        # dict. Each scoring then counts it by its own rule, exactly as for an unusable
        # response to a single attribute: class 0 in one, excluded from the other.
        error = None
        if agent_pred is None:
            error = "no_prediction"
        else:
            num_with_predictions += 1
            if agent_pred.get("image_not_found"):
                error = "image_not_found"

        if error is not None:
            all_output_predictions[segment_id] = {"error": error}
            predictions = {}
        else:
            predictions = _parse_agent_prediction(
                agent_pred, attr_to_value_to_class_idx, attrs_to_include
            )

            if any(p.pred_idx >= 0 for p in predictions.values()):
                num_valid += 1

            invalid = {a: p for a, p in predictions.items() if p.pred_idx < 0}
            if invalid:
                example = next(iter(invalid.values())).pred_value
                print(
                    f"  [INVALID] {segment_id}: {len(invalid)}/{len(predictions)} attrs"
                    f" unparsed — e.g. {example!r}"
                )

            all_output_predictions[segment_id] = {
                "predictions": predictions_to_json_serializable(predictions),
            }

        num_scored, num_invalid = count_scored_and_invalid_responses(predictions, attrs_to_include,
                                                attr_to_value_to_class_idx)
        num_scored_responses += num_scored
        num_invalid_responses += num_invalid

        if target is not None:
            t = (target if isinstance(target, torch.Tensor)
                 else torch.as_tensor(target)).unsqueeze(0)
            update_both_scorings([metrics], [metrics_excluding_invalid], [predictions], t,
                               attr_to_value_to_class_idx, attrs_order,
                               attrs_to_include=attrs_to_include)

        num_completed += 1

    computed_metrics = metrics.compute()
    computed_metrics_excluding_invalid = metrics_excluding_invalid.compute()
    predictions_out_file = output_dir / "predictions.json"

    # Built before the report so that what is printed and what is saved are read off
    # the returned object, rather than recomputed from the same counters.
    result = AgentEvaluationResult(
        num_samples_in_split=len(dataset),
        num_samples_with_predictions=num_with_predictions,
        num_samples_completed=num_completed,
        num_valid_predictions=num_valid,
        num_scored_attribute_responses=num_scored_responses,
        num_invalid_attribute_responses=num_invalid_responses,
        metrics=computed_metrics,
        metrics_excluding_invalid=computed_metrics_excluding_invalid,
        output_dir=output_dir,
        predictions_file=predictions_out_file,
    )

    print("\n=== Evaluation Metrics ===")
    coverage = num_with_predictions / len(dataset) if len(dataset) > 0 else 0.0
    print(f"  Coverage: {num_with_predictions}/{len(dataset)} segments ({coverage:.1%})")
    print_metrics("unusable responses scored as class 0", computed_metrics)
    print_metrics("unusable responses excluded", computed_metrics_excluding_invalid)
    print(f"\n  invalid responses: {num_invalid_responses}/{num_scored_responses} "
          f"({result.invalid_rate:.4f})")

    # Save outputs
    with open(predictions_out_file, "w", encoding="utf-8") as f:
        json.dump(all_output_predictions, f, indent=2, ensure_ascii=False)

    summary = {
        "num_samples_in_split": len(dataset),
        "num_samples_with_predictions": num_with_predictions,
        "num_samples_completed": num_completed,
        "num_valid_predictions": num_valid,
        "num_scored_attribute_responses": num_scored_responses,
        "num_invalid_attribute_responses": num_invalid_responses,
        "invalid_rate": result.invalid_rate,
        "coverage": coverage,
        "split": split,
        "dataset": dataset_name,
        "predictions_source": str(predictions_file),
        "metrics": metrics_to_json_dict(computed_metrics),
        "metrics_excluding_invalid": metrics_to_json_dict(computed_metrics_excluding_invalid),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  predictions.json: {len(all_output_predictions)} entries")
    print("  summary.json")

    return result
