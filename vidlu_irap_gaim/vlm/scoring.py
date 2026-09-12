"""Scoring parsed VLM predictions against ground truth, both ways at once.

An unusable response – one the model omitted, or whose value matched nothing in
the value set – can be scored two ways, and neither subsumes the other:

- as class 0, which is what the training-time eval does and so gives the
  comparable number, but generously, since class 0 is often the majority value
  ("None");
- excluded from the metrics (target set to ``IGNORE_LABEL_INDEX``), which
  measures classification skill *conditional on* a format-compliant response and has to
  be read next to the invalid rate, not on its own.

So an evaluation run does not pick one: it keeps two metric sets and updates
both from the same predictions, and reports both. The pairing lives here rather
than at the call sites because the exclusion lives in the *targets*, not in the
model outputs: a caller that scores against its own targets silently gets
"class0" under an "ignore" label. Keeping both scorings in one place makes that
unwritable.

The summary helpers at the end are shared by the evaluators that report both
scorings, so that the reports cannot drift apart.
"""

import typing as T

import torch

from irap_data import IGNORE_LABEL_INDEX
from vidlu.utils.collections import NameDict

from .predictions import attribute_predictions_to_one_hot_outputs, is_usable_prediction
from .response_parser import AttributePrediction


def update_both_scorings(
        metrics_invalid_as_class0: T.Iterable,
        metrics_invalid_ignored: T.Iterable,
        batch_predictions: list[dict[str, AttributePrediction]],
        targets: torch.Tensor,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attrs_order: T.Sequence[str],
        *,
        attrs_to_include: T.Sequence[str] | None = None,
        device: str | torch.device = "cpu",
) -> None:
    """Updates both scorings from one batch of parsed predictions.

    Args:
        metrics_invalid_as_class0: Metrics scoring an unusable response as class 0.
        metrics_invalid_ignored: Metrics excluding an unusable response instead.
        batch_predictions: Parsed predictions, one dict per sample.  An empty
            dict is a sample with no usable response at all, which each scoring
            counts by its own rule.
        targets: (B, A) ground truth.  Not mutated.
        attr_to_value_to_class_idx: Attribute metadata, for the class counts.
        attrs_order: Full attribute order, matching the dataset's.
        attrs_to_include: The attributes actually asked about; None means all.
        device: Device for the constructed output tensors.
    """
    out, is_invalid = attribute_predictions_to_one_hot_outputs(
        batch_predictions, attr_to_value_to_class_idx, attrs_order, device,
        attrs_to_include=attrs_to_include)
    # The outputs are shared: an unusable response is one-hot at class 0 in both scorings, and
    # the confusion matrix already drops IGNORE_LABEL_INDEX targets, so masking the target is
    # all the exclusion takes.
    ignore_targets = targets.masked_fill(is_invalid.to(targets.device), IGNORE_LABEL_INDEX)
    for metrics, scoring_targets in ((metrics_invalid_as_class0, targets),
                                    (metrics_invalid_ignored, ignore_targets)):
        for m in metrics:
            m.update(NameDict(out=out, target=scoring_targets))


def count_scored_and_invalid_responses(
        predictions: dict[str, AttributePrediction],
        attrs_to_include: T.Sequence[str],
        attr_to_value_to_class_idx: dict[str, dict[str, int]]) -> tuple[int, int]:
    """``(num_scored, num_invalid)`` for one sample's responses.

    Every attribute asked about counts as scored; one counts as invalid when it has
    no usable prediction (see `is_usable_prediction`).  A sample with no responses at
    all is therefore all-invalid, which is what makes the invalid rate account for
    segments the model never responded to.
    """
    num_invalid = sum(
        1 for attr in attrs_to_include
        if not is_usable_prediction(predictions.get(attr), len(attr_to_value_to_class_idx[attr])))
    return len(attrs_to_include), num_invalid


# Summary helpers ##################################################################################

def merge_computed_metrics(metrics: T.Iterable) -> dict[str, T.Any]:
    """Merges the `compute()` results of several metric objects into one dictionary."""
    computed: dict[str, T.Any] = {}
    for m in metrics:
        computed.update(m.compute())
    return computed


def print_metrics(scoring_name: str, computed: dict[str, T.Any]) -> None:
    """Prints scalar metrics, and the attribute-average of per-attribute ones."""
    print(f"\n  [{scoring}]")
    for k, v in computed.items():
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.4f}")
        elif isinstance(v, dict):
            avg = sum(v.values()) / len(v) if v else 0.0
            print(f"    {k} (avg): {avg:.4f}")


def metrics_to_json_dict(computed: dict[str, T.Any]) -> dict[str, T.Any]:
    """The computed metrics as plain scalars and dictionaries, for `json.dump`."""
    return {k: (v if isinstance(v, (int, float, type(None))) else dict(v))
            for k, v in computed.items()}
