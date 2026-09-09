import pytest

from vidlu.experiments import get_main_metric_value


def test_the_first_main_metric_is_used():
    metrics = dict(amF1=0.5, A=0.9)
    assert get_main_metric_value(metrics, ("A", "amF1"), "val") == 0.9


def test_no_main_metric_falls_back_to_the_first_produced_one():
    assert get_main_metric_value(dict(A=0.9, amF1=0.5), (), "val") == 0.9


def test_an_unproducible_main_metric_names_the_producible_ones():
    """Without this the name surfaces as a bare KeyError at the first checkpoint save.

    It is the default of `MultiAttributeClassification` when `--metrics` does not supply
    `get_irap_metrics`, and it is what a typo in `--main_metrics` looks like.
    """
    with pytest.raises(KeyError) as excinfo:
        get_main_metric_value(dict(A=0.9, loss=1.2), ("amF1",), "val")
    message = str(excinfo.value)
    assert all(s in message for s in ("amF1", "'A'", "'loss'", "--main_metrics"))
