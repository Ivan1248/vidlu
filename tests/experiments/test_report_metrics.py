"""Tests for `vidlu.experiments.report_metrics`: what goes on the console line and what
reaches the tracker."""

from vidlu.experiments import report_metrics
from vidlu.training.trainers import IterState
from vidlu.utils.logger import Logger


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_scalars(self, metrics, step, split=None):
        self.calls.append((dict(metrics), step, split))


def test_mapping_valued_metrics_are_off_the_console_line_but_reach_the_tracker():
    logger = Logger(emit=lambda _: None)
    tracker = RecordingTracker()
    metrics = {"amF1": 0.5, "per_attribute": {"a": 0.4, "b": 0.6}}
    report_metrics(IterState(batch_count=10, iteration=3, abs_iteration=13), is_training=False,
                   metrics=metrics, epoch=0, epoch_count=1, split_name="val", logger=logger,
                   tracker=tracker)
    line = logger.records[-1][1]
    assert "amF1=.5000" in line and "per_attribute" not in line
    assert tracker.calls == [(metrics, 13, "val")]


def test_the_console_filter_selects_what_goes_on_the_console_line():
    logger = Logger(emit=lambda _: None)
    metrics = {"loss": 1.0, "aux": 2.0}
    report_metrics(IterState(batch_count=10, iteration=3), is_training=True, metrics=metrics,
                   epoch=0, epoch_count=1, logger=logger,
                   console_filter=lambda k, v: k != "aux")
    line = logger.records[-1][1]
    assert "loss=" in line and "aux=" not in line
