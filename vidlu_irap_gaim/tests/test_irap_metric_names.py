"""Tests for the iRAP metric selection in `irap_metric_names`.

`get_irap_metrics` itself needs an iRAP dataset; what is iRAP-specific without one is which
metrics are requested under which result keys, in which order (the first is the fallback main
metric), and how `output_kind` changes the set.
"""

from vidlu.experiments import console_hidden, is_console_hidden
from vidlu.metrics import MultiAttributeClassificationMetrics
from vidlu_irap_gaim.metrics import IRAP_MAIN_METRIC, irap_metric_names


def test_the_main_metric_comes_first_and_every_name_parses():
    names = irap_metric_names()
    assert next(iter(names)) == "a" + IRAP_MAIN_METRIC
    # Construction parses every name; an unknown one would raise.
    MultiAttributeClassificationMetrics({"x": (0, 2)}, metrics=names)


def test_per_attribute_metrics_are_keyed_as_console_hidden_and_scalars_are_not():
    for key, name in irap_metric_names().items():
        assert is_console_hidden(key) == (key != name)
        assert is_console_hidden(key) == (not name.startswith("a")), (key, name)


def test_hard_outputs_leave_out_the_probabilistic_names():
    names = irap_metric_names(output_kind="hard")
    probabilistic = MultiAttributeClassificationMetrics.PROBABILISTIC_METRICS
    assert not any(n in probabilistic or n.startswith(("aNLL", "aBrier", "amNLL"))
                   for n in names.values())
    MultiAttributeClassificationMetrics({"x": (0, 2)}, metrics=names, output_kind="hard")
    assert {"aNLL", "aBrier", "amNLL"} <= set(irap_metric_names())


def test_each_support_threshold_adds_its_restricted_names():
    names = irap_metric_names(min_class_supports=(3,))
    assert {"amF1_supp3", console_hidden("mF1_supp3"), console_hidden("nc_supp3"),
            "amNLL_supp3"} <= set(names)
    assert not any("_supp" in n for n in irap_metric_names(min_class_supports=()))


def test_accuracy_is_averaged_over_attributes():
    names = irap_metric_names()
    assert "aA" in names and names[console_hidden("A")] == "A"
