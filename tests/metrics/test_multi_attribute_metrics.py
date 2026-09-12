"""Tests for `MultiAttributeClassificationMetrics` through `update`: attribute addressing,
`output_kind`, the probabilistic children and attribute averaging.

Synthetic logits and targets only: no model, no GPU.
"""

import math

import numpy as np
import pytest
import torch

from vidlu.metrics import AttributeSpec, MultiAttributeClassificationMetrics
from vidlu.utils.collections import NameDict

CLASS_COUNTS = {"a": 3, "b": 2}


def make_metrics(metrics, output_kind="logits", **kwargs):
    return MultiAttributeClassificationMetrics(
        {a: AttributeSpec(i, c) for i, (a, c) in enumerate(CLASS_COUNTS.items())},
        metrics=metrics, output_kind=output_kind, ignore_missing_classes=True, **kwargs)


def random_batch(num_examples=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    outs = tuple(torch.randn(num_examples, c, generator=g) for c in CLASS_COUNTS.values())
    target = torch.stack([torch.randint(c, (num_examples,), generator=g)
                          for c in CLASS_COUNTS.values()], dim=1)
    return NameDict(out=outs, target=target)


# Attribute addressing ###########################################################

def test_attribute_keys_are_the_result_keys_and_indices_select_the_outputs():
    batch = random_batch()
    # Keys and output positions are independent: 'second' reads output 1, 'first' output 0.
    m = MultiAttributeClassificationMetrics(
        {"second": (1, 2), "first": AttributeSpec(0, 3)}, metrics=("A", "n"))
    m.update(batch)
    r = m.compute()
    assert list(r["A"]) == ["second", "first"]
    expected_second = (batch.out[1].argmax(1) == batch.target[:, 1]).double().mean().item()
    assert r["A"]["second"] == pytest.approx(expected_second)
    assert r["n"]["first"] == 64


def test_integer_keys_work():
    m = MultiAttributeClassificationMetrics({0: (0, 3), 1: (1, 2)}, metrics=("A", "aA"))
    m.update(random_batch())
    assert set(m.compute()["A"]) == {0, 1}


def test_a_mapping_gives_each_metric_the_callers_result_key():
    # The metric knows nothing about how a key is reported; e.g. the reporting layer's
    # console-hidden keys are just keys that differ from the metric name.
    m = make_metrics({"per_attribute_A": "A", "aA": "aA"})
    m.update(random_batch())
    r = m.compute()
    assert set(r) == {"per_attribute_A", "aA"}
    assert r["aA"] == pytest.approx(np.mean(list(r["per_attribute_A"].values())))
    assert set(m.compute(metrics={"acc": "A"})) == {"acc"}
    assert set(m.compute(metrics=("A",))) == {"A"}


def test_an_output_index_beyond_the_outputs_raises_instead_of_being_skipped():
    m = MultiAttributeClassificationMetrics({"a": (0, 3), "late": (5, 2)}, metrics=("aA",))
    with pytest.raises(ValueError, match="'late'.*index 5.*2 outputs"):
        m.update(random_batch())


def test_get_outputs_can_unbind_a_stacked_tensor():
    batch = random_batch()
    stacked = torch.stack([batch.out[0], torch.nn.functional.pad(batch.out[1], (0, 1))], dim=1)
    # Attribute 'b' has 2 classes but gets a padded third column here, so give it 3.
    m = MultiAttributeClassificationMetrics(
        {"a": (0, 3), "b": (1, 3)}, metrics=("A",), get_outputs=lambda r: r.out.unbind(1))
    m.update(NameDict(out=stacked, target=batch.target))
    reference = make_metrics(("A",))
    reference.update(batch)
    assert m.compute()["A"]["a"] == pytest.approx(reference.compute()["A"]["a"])


def test_none_outputs_skip_the_update():
    m = make_metrics(("n",))
    m.update(NameDict(out=None, target=None))
    assert all(v == 0 for v in m.compute()["n"].values())


def test_ignore_index_targets_are_excluded_from_every_child():
    batch = random_batch()
    target = batch.target.clone()
    target[::2, 0] = -1
    m = make_metrics(("n", "NLL"))
    m.update(NameDict(out=batch.out, target=target))
    r = m.compute()
    assert r["n"]["a"] == 32 and r["n"]["b"] == 64
    kept = target[:, 0] != -1
    expected = -batch.out[0][kept].double().log_softmax(1).gather(
        1, target[kept, 0:1]).mean().item()
    assert r["NLL"]["a"] == pytest.approx(expected)


def test_at_least_one_attribute_is_required():
    with pytest.raises(ValueError, match="At least one attribute"):
        MultiAttributeClassificationMetrics({}, metrics=("aA",))


# output_kind and the probabilistic children ######################################

def test_hard_outputs_refuse_probabilistic_metrics_and_accept_the_rest():
    with pytest.raises(ValueError, match="output_kind='hard'"):
        make_metrics(("aNLL",), output_kind="hard")
    m = make_metrics(("amF1", "aMCC", "MCC"), output_kind="hard")
    assert m.attr_to_prob_metrics == {}
    m.update(random_batch())
    assert set(m.compute()) == {"amF1", "aMCC", "MCC"}


def test_probabilistic_children_exist_only_when_needed():
    assert make_metrics(("amF1",)).attr_to_prob_metrics == {}
    assert set(make_metrics(("amF1", "aBrier")).attr_to_prob_metrics) == set(CLASS_COUNTS)


def test_computing_an_unaccumulated_probabilistic_metric_raises():
    m = make_metrics(("amF1",))
    m.update(random_batch())
    with pytest.raises(ValueError, match="none was accumulated"):
        m.compute(metrics=("aNLL",))


def test_attribute_average_of_nll_is_the_mean_of_per_attribute_nll():
    batch = random_batch()
    m = make_metrics(("aNLL", "NLL", "aBrier", "Brier", "amNLL", "mNLL"))
    m.update(batch)
    r = m.compute()
    for i, a in enumerate(CLASS_COUNTS):
        expected = -batch.out[i].double().log_softmax(1).gather(
            1, batch.target[:, i:i + 1]).mean().item()
        assert r["NLL"][a] == pytest.approx(expected)
        assert 0.0 <= r["Brier"][a] <= 2.0
    assert r["aNLL"] == pytest.approx(np.mean(list(r["NLL"].values())))
    assert r["aBrier"] == pytest.approx(np.mean(list(r["Brier"].values())))
    assert r["amNLL"] == pytest.approx(np.mean(list(r["mNLL"].values())))


def test_probs_kind_matches_logits_kind_on_softmaxed_outputs():
    batch = random_batch()
    from_logits = make_metrics(("aNLL", "aBrier", "amNLL"))
    from_probs = make_metrics(("aNLL", "aBrier", "amNLL"), output_kind="probs")
    from_logits.update(batch)
    from_probs.update(NameDict(out=tuple(o.softmax(1) for o in batch.out), target=batch.target))
    assert from_probs.compute() == pytest.approx(from_logits.compute(), rel=1e-6)


def test_restricted_class_balanced_nll_uses_ground_truth_support():
    # Attribute 'a': class 2 has 2 ground-truth examples, the others 31 each, so `_supp10`
    # excludes exactly class 2 while `_supp1` keeps every class.
    batch = random_batch()
    batch.target[:, 0] = torch.tensor([0, 1] * 31 + [2, 2])
    m = make_metrics(("mNLL", "mNLL_supp1", "mNLL_supp10", "nc_supp10", "cNLL"))
    m.update(batch)
    r = m.compute()
    per_class = r["cNLL"]["a"]
    assert r["nc_supp10"]["a"] == 2
    assert r["mNLL_supp1"]["a"] == pytest.approx(r["mNLL"]["a"])
    assert r["mNLL_supp10"]["a"] == pytest.approx(float(np.mean(per_class[:2])))


# Attribute averaging ############################################################

def test_undefined_attribute_is_dropped_from_the_average():
    # Attribute 'b' has a single ground-truth label everywhere and a single prediction, so
    # its MCC is undefined (NaN); the average must be attribute 'a' alone.
    batch = random_batch()
    target = batch.target.clone()
    target[:, 1] = 0
    out = (batch.out[0], torch.tensor([[1.0, 0.0]]).expand(len(target), 2))
    m = make_metrics(("aMCC", "MCC"))
    m.update(NameDict(out=out, target=target))
    r = m.compute()
    assert math.isnan(r["MCC"]["b"])
    assert r["aMCC"] == pytest.approx(r["MCC"]["a"])


def test_averaging_a_per_class_array_raises():
    m = make_metrics(("aF1",))
    m.update(random_batch())
    with pytest.raises(ValueError, match="Cannot average per-class"):
        m.compute()


def test_confusion_matrices_are_exposed_per_attribute():
    batch = random_batch()
    m = make_metrics(("aA",))
    m.update(batch)
    cms = m.get_confusion_matrices()
    assert set(cms) == set(CLASS_COUNTS)
    assert cms["a"].shape == (3, 3) and cms["a"].sum() == 64
    assert cms["a"].sum(1).tolist() == torch.bincount(batch.target[:, 0], minlength=3).tolist()


def test_reset_clears_every_child():
    m = make_metrics(("aNLL", "n"))
    m.update(random_batch())
    m.reset()
    assert all(v == 0 for v in m.compute()["n"].values())
    assert all(p.count.sum() == 0 for p in m.attr_to_prob_metrics.values())
