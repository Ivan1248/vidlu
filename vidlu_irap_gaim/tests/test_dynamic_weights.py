"""Tests for the dynamic balanced-recall class weights.

Datasets, the trainer and the metrics are all stubbed, so these run without IRAP_HOME and
without touching a model. What they pin down is what the port of the original
`calculate_new_class_weights` got wrong: which examples are counted (the ignore label),
which splits they come from (all `train*`, not the first), and which split's confusion
matrix the recalls are read from.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vidlu_irap_gaim.training.dynamic_weights import (
    DynamicBalancedRecallWeights, compute_attr_to_class_occurrence_counts,
    _calculate_attr_to_class_weights, INVERSE_FREQUENCY_FOR_ABSENT_CLASS)

IGNORE = -1


class _StubDataset:
    """A stub carrying just the `info` fields the weighting reads.

    Mirrors the stub in `test_attribute_distribution_report.py`, plus `__len__`: the
    extension checks it against `len(info.segment_ids)` to catch joined datasets, whose
    `info` describes only one of the joined parts.
    """

    def __init__(self, info, num_examples):
        self.info = info
        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples


def _dataset(labels_by_segment, *, segment_ids, class_counts, attr_values):
    return _StubDataset(
        SimpleNamespace(
            segment_ids=segment_ids,
            segment_id_to_labels=labels_by_segment,
            class_counts=class_counts,
            attr_to_value_to_class_idx={
                attr: {name: i for i, name in enumerate(names)}
                for attr, names in attr_values.items()}),
        num_examples=len(segment_ids))


def _one_attribute_dataset(labels, *, num_classes, num_described=None):
    """A single-attribute dataset whose segment `i` has label `labels[i]`."""
    segment_ids = [f"s{i}" for i in range(len(labels))]
    dataset = _dataset({sid: [v] for sid, v in zip(segment_ids, labels)},
                       segment_ids=segment_ids, class_counts=(num_classes,),
                       attr_values={"Attr": tuple("abcdefgh")[:num_classes]})
    if num_described is not None:  # simulate a joined or subsetted dataset
        dataset.info.segment_ids = segment_ids[:num_described]
    return dataset


class _StubMetric:
    """A metric whose confusion matrices are built from per-class (tp, actual) counts."""

    def __init__(self, attr_to_true_positives_and_actual):
        self.cms = {attr: self._confusion_matrix(tp, actual)
                    for attr, (tp, actual) in attr_to_true_positives_and_actual.items()}

    @staticmethod
    def _confusion_matrix(tp, actual):
        tp, actual = torch.tensor(tp), torch.tensor(actual)
        cm = torch.diag(tp)
        for i, misses in enumerate((actual - tp).tolist()):  # rows must sum to `actual`
            cm[i, (i + 1) % len(tp)] += misses
        return cm

    def get_confusion_matrices(self):
        return self.cms

    def reset(self):
        self.cms = {attr: torch.zeros_like(cm) for attr, cm in self.cms.items()}


class _StubLoss:
    supports_dynamic_class_weights = True

    def __init__(self):
        self.attrs_idx = None
        self.attr_idx_to_class_weights = None

    def set_attrs_idx(self, attrs_idx):
        self.attrs_idx = list(attrs_idx)

    def set_class_weights(self, attr_idx_to_class_weights):
        self.attr_idx_to_class_weights = attr_idx_to_class_weights


class _StubEvent:
    def __init__(self):
        self.handlers = []

    def handler(self, f):
        self.handlers.append(f)
        return f

    def fire(self, split_name):
        for f in self.handlers:
            f(SimpleNamespace(split_name=split_name))


class _StubTrainer:
    """Enough of `Trainer` for the extension: `data`, `loss` and `get_metrics`."""

    def __init__(self, data, metrics):
        self.data = data
        self.metrics = metrics  # a list, or a {split_name: [metric]} mapping
        self.loss = _StubLoss()
        self.evaluation = SimpleNamespace(epoch_completed=_StubEvent())

    def get_metrics(self, split_name=None):
        if not isinstance(self.metrics, dict):
            return self.metrics
        return self.metrics.get(split_name, next(iter(self.metrics.values())))


def _initialized(data, metrics, **kwargs):
    extension = DynamicBalancedRecallWeights(attrs_to_include=("Attr",), **kwargs)
    trainer = _StubTrainer(data, metrics)
    extension.initialize(trainer)
    return extension, trainer


# Counting ########################################################################

def test_ignore_label_is_excluded_not_counted_as_a_class():
    """Vietnam leaves 7 attributes unannotated as -1. `counts[-1] += 1` would credit
    every one of those examples to the *last* class and inflate the total."""
    dataset = _one_attribute_dataset([IGNORE] * 50 + [0, 2], num_classes=3)
    counts = compute_attr_to_class_occurrence_counts([dataset], {"Attr": 0}, {"Attr": 3})
    assert counts["Attr"].tolist() == [1, 0, 1]  # not [1, 0, 51]


def test_counts_only_the_examples_the_split_yields():
    """`segment_id_to_labels` is built before context filtering, so it covers more
    segments than the split."""
    dataset = _one_attribute_dataset([0, 0, 1, 1], num_classes=2)
    dataset.info.segment_ids = ["s0", "s1", "s2"]  # 's3' is in the mapping, not the split
    counts = compute_attr_to_class_occurrence_counts([dataset], {"Attr": 0}, {"Attr": 2})
    assert counts["Attr"].tolist() == [2, 1]


def test_counts_are_pooled_over_every_train_split():
    bih = _one_attribute_dataset([0, 0, 0], num_classes=2)
    vietnam = _one_attribute_dataset([1, 1], num_classes=2)
    pooled = compute_attr_to_class_occurrence_counts([bih, vietnam], {"Attr": 0}, {"Attr": 2})
    assert pooled["Attr"].tolist() == [3, 2]
    # Taking only the first split -- the old behaviour -- gives a different, wrong answer.
    assert compute_attr_to_class_occurrence_counts(
        [bih], {"Attr": 0}, {"Attr": 2})["Attr"].tolist() == [3, 0]


def test_class_index_outside_the_range_raises():
    dataset = _one_attribute_dataset([0, 5], num_classes=2)
    with pytest.raises(ValueError, match="outside"):
        compute_attr_to_class_occurrence_counts([dataset], {"Attr": 0}, {"Attr": 2})


def test_extension_pools_priors_over_every_train_split():
    extension, _ = _initialized(
        dict(train_bih=_one_attribute_dataset([0, 0, 0], num_classes=2),
             train_vn=_one_attribute_dataset([1, 1], num_classes=2),
             val=_one_attribute_dataset([0, 1], num_classes=2)),
        [_StubMetric({"Attr": ([1, 1], [1, 1])})])
    assert extension.attr_to_class_occurrence_counts["Attr"].tolist() == [3, 2]


def test_a_dataset_whose_info_does_not_describe_it_raises():
    """`a.join(b, info=b.info)` keeps one info, so the priors would cover one domain."""
    data = dict(train=_one_attribute_dataset([0, 1, 0, 1], num_classes=2, num_described=2),
                val=_one_attribute_dataset([0, 1], num_classes=2))
    with pytest.raises(ValueError, match="info describes"):
        _initialized(data, [_StubMetric({"Attr": ([1, 1], [1, 1])})])


# Weights #########################################################################

def test_weights_match_the_original_formula():
    """`w = inv_freq*(1-R) + sqrt(inv_freq)*R`, from train_local_rec.py:154-162."""
    counts = {"Attr": torch.tensor([3, 1])}  # total 4 -> inv_freq 4/3, 4
    recalls = {"Attr": np.array([1.0, 0.0])}
    weights = _calculate_attr_to_class_weights(counts, recalls)["Attr"]
    assert weights[0].item() == pytest.approx(math.sqrt(4 / 3))  # R=1 -> sqrt(inv_freq)
    assert weights[1].item() == pytest.approx(4.0)               # R=0 -> inv_freq


def test_a_class_absent_from_training_gets_the_original_constant():
    counts = {"Attr": torch.tensor([4, 0])}
    weights = _calculate_attr_to_class_weights(counts, {"Attr": np.array([0.0, 0.0])})["Attr"]
    assert weights[1].item() == pytest.approx(INVERSE_FREQUENCY_FOR_ABSENT_CLASS)


def test_without_recalls_a_random_classifier_is_assumed():
    """The original bootstraps epoch 0 with recalls of 1/num_classes."""
    counts = {"Attr": torch.tensor([3, 1])}
    weights = _calculate_attr_to_class_weights(counts)["Attr"]
    expected = [f * (1 - 0.5) + math.sqrt(f) * 0.5 for f in (4 / 3, 4.0)]
    assert weights.tolist() == pytest.approx(expected)


def test_totals_exclude_ignored_examples():
    """The inverse frequencies are taken over labelled examples, not over all segments."""
    dataset = _one_attribute_dataset([IGNORE] * 96 + [0, 0, 0, 1], num_classes=2)
    counts = compute_attr_to_class_occurrence_counts([dataset], {"Attr": 0}, {"Attr": 2})
    weights = _calculate_attr_to_class_weights(counts, {"Attr": np.array([0.0, 0.0])})["Attr"]
    assert weights.tolist() == pytest.approx([4 / 3, 4.0])  # total 4, not 100


def test_an_attribute_with_no_labelled_example_is_not_weighted():
    extension, trainer = _initialized(
        dict(train=_one_attribute_dataset([IGNORE, IGNORE], num_classes=2),
             train_ok=_one_attribute_dataset([0, 1], num_classes=2),
             val=_one_attribute_dataset([0, 1], num_classes=2)),
        [_StubMetric({"Attr": ([1, 1], [1, 1])})])
    assert extension.attr_to_class_occurrence_counts["Attr"].tolist() == [1, 1]

    all_ignored = dict(train=_one_attribute_dataset([IGNORE, IGNORE], num_classes=2),
                       val=_one_attribute_dataset([0, 1], num_classes=2))
    with pytest.raises(ValueError, match="has a labelled training example"):
        _initialized(all_ignored, [_StubMetric({"Attr": ([1, 1], [1, 1])})])


# Recalls: which split's confusion matrix is read ##################################

def _data_with_two_val_splits():
    return dict(train=_one_attribute_dataset([0, 0, 0, 1], num_classes=2),
                val_a=_one_attribute_dataset([0, 1], num_classes=2),
                val_b=_one_attribute_dataset([0, 1], num_classes=2))


def test_recalls_come_from_the_evaluated_splits_own_metric():
    """`get_metrics()` without a split name returns the *first* entry of a per-split
    metrics mapping, whose confusion matrix belongs to another split."""
    val_a = _StubMetric({"Attr": ([0, 0], [4, 4])})  # recall 0 -> weights = inv_freq
    val_b = _StubMetric({"Attr": ([4, 4], [4, 4])})  # recall 1 -> weights = sqrt(inv_freq)
    extension, trainer = _initialized(_data_with_two_val_splits(),
                                      dict(val_a=[val_a], val_b=[val_b]),
                                      recall_split_names=["val_b"])
    trainer.evaluation.epoch_completed.fire("val_b")
    weights = extension.attr_to_class_weights["Attr"]
    assert weights.tolist() == pytest.approx([math.sqrt(4 / 3), 2.0])  # val_b's, not val_a's


def test_a_reset_metric_of_another_split_cannot_drive_the_weights():
    """The failure this replaces: the handler fired for every `val*` split but always read
    the first split's metric, which had already been reset -- `actual == 0` everywhere,
    hence recall 1 everywhere, collapsing the weights to sqrt(inv_freq) every epoch."""
    val_a = _StubMetric({"Attr": ([0, 0], [4, 4])})
    val_b = _StubMetric({"Attr": ([0, 0], [4, 4])})
    extension, trainer = _initialized(_data_with_two_val_splits(),
                                      dict(val_a=[val_a], val_b=[val_b]))
    assert extension.target_split_names == ["val_a"]  # only the first, by default

    trainer.evaluation.epoch_completed.fire("val_a")
    from_val_a = extension.attr_to_class_weights["Attr"].clone()
    val_a.reset()  # what ProgressMonitor does after reporting the split
    trainer.evaluation.epoch_completed.fire("val_b")  # not a target split: ignored

    assert extension.attr_to_class_weights["Attr"].tolist() == from_val_a.tolist()
    assert from_val_a.tolist() == pytest.approx([4 / 3, 4.0])  # recall 0, not the sqrt collapse


def test_naming_several_recall_splits_pools_their_statistics():
    val_a = _StubMetric({"Attr": ([0, 0], [4, 0])})  # 0/4 on class 0, nothing on class 1
    val_b = _StubMetric({"Attr": ([0, 4], [0, 4])})  # nothing on class 0, 4/4 on class 1
    extension, trainer = _initialized(_data_with_two_val_splits(),
                                      dict(val_a=[val_a], val_b=[val_b]),
                                      recall_split_names=["val_a", "val_b"])
    trainer.evaluation.epoch_completed.fire("val_a")
    trainer.evaluation.epoch_completed.fire("val_b")
    # Pooled: class 0 -> 0/4 = 0, class 1 -> 4/4 = 1.
    assert extension.attr_to_class_weights["Attr"].tolist() == pytest.approx([4 / 3, 2.0])


def test_pooling_restarts_each_epoch():
    val_a = _StubMetric({"Attr": ([0, 0], [4, 4])})
    val_b = _StubMetric({"Attr": ([0, 0], [4, 4])})
    extension, trainer = _initialized(_data_with_two_val_splits(),
                                      dict(val_a=[val_a], val_b=[val_b]),
                                      recall_split_names=["val_a", "val_b"])
    for _ in range(2):
        trainer.evaluation.epoch_completed.fire("val_a")
        trainer.evaluation.epoch_completed.fire("val_b")
    # Without a per-epoch reset, `actual` would accumulate across epochs; recalls are
    # ratios so they stay 0 either way, but the pooled counts must not grow unboundedly.
    assert extension._pooled_attr_to_class_stats["Attr"]["actual"].tolist() == [8, 8]


def test_a_missing_validation_split_raises():
    data = dict(train=_one_attribute_dataset([0, 1], num_classes=2))
    with pytest.raises(ValueError, match="no validation split"):
        _initialized(data, [_StubMetric({"Attr": ([1, 1], [1, 1])})])


def test_a_missing_training_split_raises():
    data = dict(val=_one_attribute_dataset([0, 1], num_classes=2))
    with pytest.raises(ValueError, match="no training split"):
        _initialized(data, [_StubMetric({"Attr": ([1, 1], [1, 1])})])


# Loss wiring and checkpointing ####################################################

def test_weights_reach_the_loss_keyed_by_global_attribute_index():
    extension, trainer = _initialized(
        dict(train=_one_attribute_dataset([0, 0, 0, 1], num_classes=2),
             val=_one_attribute_dataset([0, 1], num_classes=2)),
        [_StubMetric({"Attr": ([0, 0], [4, 4])})])
    assert trainer.loss.attrs_idx == [0]
    trainer.evaluation.epoch_completed.fire("val")
    assert trainer.loss.attr_idx_to_class_weights[0].tolist() == pytest.approx([4 / 3, 4.0])


def test_state_dict_carries_weights_but_recomputes_counts():
    """Occurrence counts must not be restored: a checkpoint written before this fix holds
    counts that credited the ignore label to the last class."""
    data = dict(train=_one_attribute_dataset([0, 0, 0, 1], num_classes=2),
                val=_one_attribute_dataset([0, 1], num_classes=2))
    saved, trainer = _initialized(data, [_StubMetric({"Attr": ([0, 0], [4, 4])})])
    trainer.evaluation.epoch_completed.fire("val")
    state = saved.state_dict()
    assert "attr_to_class_occurrence_counts" not in state

    restored, restored_trainer = _initialized(data, [_StubMetric({"Attr": ([1, 1], [1, 1])})])
    restored.load_state_dict(state)
    assert restored.attr_to_class_weights["Attr"].tolist() == pytest.approx([4 / 3, 4.0])
    assert restored.attr_to_class_occurrence_counts["Attr"].tolist() == [3, 1]
    assert restored_trainer.loss.attr_idx_to_class_weights[0].tolist() == pytest.approx(
        [4 / 3, 4.0])


def test_a_pre_fix_checkpoint_is_rejected_rather_than_silently_reused():
    data = dict(train=_one_attribute_dataset([0, 0, 0, 1], num_classes=2),
                val=_one_attribute_dataset([0, 1], num_classes=2))
    extension, _ = _initialized(data, [_StubMetric({"Attr": ([0, 0], [4, 4])})])
    old_state = {"attr_key_to_class_weights": {"Attr": torch.tensor([1.0, 1.0])},
                 "attr_key_to_class_occurrence_counts": {"Attr": torch.tensor([3, 49])}}
    with pytest.raises(KeyError, match="predates"):
        extension.load_state_dict(old_state)


def test_a_loss_without_class_weight_support_raises():
    extension = DynamicBalancedRecallWeights(attrs_to_include=("Attr",))
    trainer = _StubTrainer(dict(train=_one_attribute_dataset([0, 1], num_classes=2),
                                val=_one_attribute_dataset([0, 1], num_classes=2)),
                           [_StubMetric({"Attr": ([1, 1], [1, 1])})])
    trainer.loss = object()
    with pytest.raises(TypeError, match="set_attrs_idx"):
        extension.initialize(trainer)
