"""Tests for the support-restricted (`_suppN`) metric names and the multi-attribute grammar.

Everything is driven by synthetic confusion matrices written straight into the per-attribute
`ClassificationMetrics`, so these run without a model and without a GPU.

What they pin down is the part that makes the metric usable for model comparison:

- the restricted class set comes from *ground-truth* support only, so it does not move when
  the predictions do (`test_class_subset_does_not_depend_on_predictions`);
- `mF1` and `mF1_supp1` differ exactly by the zero-support classes that the model predicted at
  least once, which is the cost that `ignore_missing_classes`'s `actual + fp > 0` mask lets
  back in (`test_absent_class_penalty_is_isolated_by_supp1`);
- an attribute with no qualifying class is dropped from the average rather than counted as
  zero (`test_attribute_without_qualifying_class_is_excluded`).
"""

import math

import numpy as np
import pytest
import torch

from vidlu.metrics import (
    MACRO_METRIC_TO_PER_CLASS,
    ClassificationMetrics,
    MultiAttributeClassificationMetrics,
    macro_over_supported,
    parse_metric_name,
)

KNOWN = MultiAttributeClassificationMetrics.KNOWN_BASE_METRICS


def make_metrics(attr_to_cm, metrics):
    """Builds the metric with confusion matrices set directly, bypassing `update`."""
    m = MultiAttributeClassificationMetrics(
        {a: (i, cm.shape[0]) for i, (a, cm) in enumerate(attr_to_cm.items())},
        metrics=metrics, ignore_missing_classes=True)
    for a, cm in attr_to_cm.items():
        m.attr_to_cm_metrics[a].cm = torch.as_tensor(cm, dtype=torch.int64)
    return m


# Name parsing ####################################################################

@pytest.mark.parametrize("name,expected", [
    ("mF1", (False, "mF1", "mF1", None)),
    ("amF1", (True, "mF1", "mF1", None)),
    ("amF1_supp10", (True, "mF1_supp10", "mF1", 10)),
    ("mF1_supp1", (False, "mF1_supp1", "mF1", 1)),
    ("nc_supp10", (False, "nc_supp10", "nc", 10)),
    ("A", (False, "A", "A", None)),
    ("aA", (True, "A", "A", None)),
    ("amNLL_supp10", (True, "mNLL_supp10", "mNLL", 10)),
])
def test_parse_metric_name(name, expected):
    assert tuple(parse_metric_name(name, KNOWN)) == expected


def test_parse_metric_name_raises_for_unknown_base():
    # Silently ignoring these would look like a metric that never fired.
    with pytest.raises(ValueError, match="Unknown metric"):
        parse_metric_name("bogus", KNOWN)


@pytest.mark.parametrize("name", ["A_supp10", "aA_supp10", "n_supp10", "aNLL_supp10"])
def test_threshold_on_non_macro_metric_raises(name):
    with pytest.raises(ValueError, match="not a mean over"):
        parse_metric_name(name, KNOWN)


def test_threshold_only_metric_without_threshold_raises():
    with pytest.raises(ValueError, match="only with a support threshold"):
        parse_metric_name("nc", KNOWN)


def test_non_integer_threshold_raises():
    with pytest.raises(ValueError, match="non-integer support threshold"):
        parse_metric_name("amF1_suppten", KNOWN)


def test_construction_rejects_a_misapplied_threshold():
    with pytest.raises(ValueError, match="not a mean over"):
        make_metrics({"a": np.eye(2, dtype=np.int64)}, metrics=("aA_supp10",))


# macro_over_supported ############################################################

def test_macro_over_supported_restricts_to_supported_classes():
    assert macro_over_supported([1.0, 0.0, 0.5], [10, 0, 10], 1).item() == 0.75


def test_macro_over_supported_at_zero_keeps_every_class():
    assert macro_over_supported([1.0, 0.0, 0.5], [10, 0, 10], 0).item() == 0.5


def test_macro_over_supported_is_nan_when_nothing_qualifies():
    assert math.isnan(macro_over_supported([1.0, 0.5], [2, 3], 10).item())


def test_macro_over_supported_handles_a_batch_dimension():
    values = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    support = torch.tensor([[10, 0], [10, 10]])
    assert macro_over_supported(values, support, 1).tolist() == [1.0, 0.5]


# Restricted means on a single ClassificationMetrics ###############################

def test_classification_metrics_computes_restricted_names_directly():
    cm = torch.tensor([[8, 1, 1],
                       [1, 8, 1],
                       [0, 0, 0]])
    m = ClassificationMetrics(class_count=3, metrics=("F1", "mF1_supp1", "nc_supp1", "nc_supp9"),
                              cm=cm, ignore_missing_classes=True)
    r = m.compute()
    assert r["nc_supp1"] == 2 and r["nc_supp9"] == 2
    assert r["mF1_supp1"] == pytest.approx(float(np.mean(r["F1"][:2])))


def test_classification_metrics_rejects_an_unknown_name():
    m = ClassificationMetrics(class_count=2, metrics=("bogus",))
    with pytest.raises(ValueError, match="Unknown metric"):
        m.compute()


# Restricted means over confusion matrices #########################################

def test_supp0_matches_the_plain_mean_over_all_classes():
    # Class 2 has no ground truth and is never predicted, so it is masked out of `mF1` but
    # still counted (as 0) by `_supp0`, which restricts nothing.
    cm = np.array([[8, 2, 0],
                   [1, 9, 0],
                   [0, 0, 0]], dtype=np.int64)
    r = make_metrics({"a": cm}, metrics=("F1", "mF1", "mF1_supp0")).compute()
    assert r["mF1_supp0"]["a"] == pytest.approx(float(np.mean(r["F1"]["a"])))
    # The unrestricted `mF1` drops the empty class, so it is the larger of the two.
    assert r["mF1"]["a"] > r["mF1_supp0"]["a"]


def test_absent_never_predicted_class_is_excluded_by_both():
    cm = np.array([[8, 2, 0],
                   [1, 9, 0],
                   [0, 0, 0]], dtype=np.int64)
    r = make_metrics({"a": cm}, metrics=("mF1", "mF1_supp1", "nc_supp1")).compute()
    assert r["nc_supp1"]["a"] == 2
    assert r["mF1_supp1"]["a"] == pytest.approx(r["mF1"]["a"])


def test_absent_class_penalty_is_isolated_by_supp1():
    # Class 2 has no ground truth but is predicted twice, so `actual + fp > 0` re-admits it
    # with F1 = 0. `_supp1` uses ground-truth support only and leaves it out.
    cm = np.array([[8, 1, 1],
                   [1, 8, 1],
                   [0, 0, 0]], dtype=np.int64)
    r = make_metrics({"a": cm}, metrics=("F1", "mF1", "mF1_supp1", "nc_supp1")).compute()
    per_class = r["F1"]["a"]
    assert per_class[2] == 0.0
    assert r["nc_supp1"]["a"] == 2
    assert r["mF1"]["a"] == pytest.approx(float(np.mean(per_class)))       # 3 classes
    assert r["mF1_supp1"]["a"] == pytest.approx(float(np.mean(per_class[:2])))  # 2 classes
    assert r["mF1_supp1"]["a"] > r["mF1"]["a"]


def test_class_subset_does_not_depend_on_predictions():
    # Same ground-truth row sums, very different predictions: `nc_suppN` must not move.
    good = np.array([[10, 0], [0, 5]], dtype=np.int64)
    bad = np.array([[0, 10], [5, 0]], dtype=np.int64)
    counts = [make_metrics({"a": cm}, metrics=("nc_supp5",)).compute()["nc_supp5"]["a"]
              for cm in (good, bad)]
    assert counts[0] == counts[1] == 2


def test_nc_counts_classes_at_or_above_the_threshold():
    cm = np.array([[10, 0, 0, 0],
                   [0, 4, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0]], dtype=np.int64)
    names = tuple(f"nc_supp{n}" for n in (1, 2, 10, 11))
    r = make_metrics({"a": cm}, metrics=names).compute()
    assert [r[name]["a"] for name in names] == [3, 2, 1, 0]


def test_attribute_without_qualifying_class_is_excluded():
    # 'big' qualifies at _supp10, 'small' does not. The average must be 'big' alone, not
    # (big + 0) / 2.
    big = np.array([[20, 0], [0, 20]], dtype=np.int64)
    small = np.array([[1, 0], [0, 1]], dtype=np.int64)
    r = make_metrics({"big": big, "small": small},
                     metrics=("amF1_supp10", "mF1_supp10", "nc_supp10")).compute()
    assert math.isnan(r["mF1_supp10"]["small"])
    assert r["nc_supp10"]["small"] == 0
    assert r["amF1_supp10"] == pytest.approx(r["mF1_supp10"]["big"])


def test_average_is_nan_when_no_attribute_qualifies():
    small = np.array([[1, 0], [0, 1]], dtype=np.int64)
    r = make_metrics({"a": small, "b": small}, metrics=("amF1_supp10",)).compute()
    assert math.isnan(r["amF1_supp10"])


def test_unrestricted_metrics_are_unchanged_by_the_new_names():
    cm = np.array([[8, 1, 1],
                   [1, 8, 1],
                   [0, 0, 0]], dtype=np.int64)
    without = make_metrics({"a": cm}, metrics=("amF1", "mF1")).compute()
    with_ = make_metrics({"a": cm}, metrics=("amF1", "mF1", "amF1_supp10", "nc_supp10")).compute()
    assert with_["amF1"] == pytest.approx(without["amF1"])
    assert with_["mF1"]["a"] == pytest.approx(without["mF1"]["a"])


def test_compute_accepts_names_outside_the_configured_metrics():
    cm = np.array([[8, 2], [1, 9]], dtype=np.int64)
    m = make_metrics({"a": cm}, metrics=("amF1",))
    # Anything the confusion matrix supports can be asked for after the fact.
    assert m.compute(metrics=("mF1_supp1", "F1"))["mF1_supp1"]["a"] == pytest.approx(
        float(np.mean(m.compute(metrics=("F1",))["F1"]["a"])))


def test_thresholds_apply_to_every_confusion_matrix_macro_metric():
    cm = np.array([[8, 1, 1],
                   [1, 8, 1],
                   [0, 0, 0]], dtype=np.int64)
    names = tuple(f"_{base}_supp1" for base, per_class in MACRO_METRIC_TO_PER_CLASS.items()
                  if per_class in MultiAttributeClassificationMetrics.CONFUSION_MATRIX_METRICS)
    r = make_metrics({"a": cm}, metrics=names).compute()
    assert set(r) == set(names)
    assert all(0.0 <= r[n]["a"] <= 1.0 for n in names)
