"""Tests for `ProbabilisticClassificationMetrics` and the chance-corrected `MCC`/`kappa`.

Reference values come from scikit-learn where it defines the same quantity (`log_loss`,
`matthews_corrcoef`, `cohen_kappa_score`) and from direct formulas otherwise (the multi-class
Brier score, which sklearn only has for the binary case).
"""

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score, log_loss, matthews_corrcoef

from vidlu.metrics import (
    ClassificationMetrics,
    ProbabilisticClassificationMetrics,
    classification_metrics,
    multiclass_confusion_matrix,
)
from vidlu.utils.collections import NameDict


def random_logits_and_targets(num_examples=200, class_count=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_examples, class_count, generator=g) * 2
    targets = torch.randint(class_count, (num_examples,), generator=g)
    return logits, targets


def brier_reference(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    one_hot = np.eye(probs.shape[1])[targets]
    return ((probs - one_hot) ** 2).sum(1)


# ProbabilisticClassificationMetrics ################################################

def test_nll_and_brier_match_the_references():
    logits, targets = random_logits_and_targets()
    probs = logits.softmax(1).double().numpy()
    m = ProbabilisticClassificationMetrics(class_count=5, output_kind='logits',
                                           metrics=('NLL', 'Brier', 'n'))
    # Two batches, to exercise accumulation.
    for sl in (slice(0, 70), slice(70, None)):
        m.update(NameDict(out=logits[sl], target=targets[sl]))
    r = m.compute()
    assert r['n'] == 200
    assert r['NLL'] == pytest.approx(log_loss(targets.numpy(), probs, labels=range(5)))
    assert r['Brier'] == pytest.approx(brier_reference(probs, targets.numpy()).mean())


def test_probs_and_logits_kinds_agree():
    logits, targets = random_logits_and_targets()
    from_logits = ProbabilisticClassificationMetrics(5, 'logits')
    from_probs = ProbabilisticClassificationMetrics(5, 'probs')
    from_logits.update(NameDict(out=logits, target=targets))
    from_probs.update(NameDict(out=logits.softmax(1), target=targets))
    for k, v in from_logits.compute().items():
        assert from_probs.compute()[k] == pytest.approx(v, rel=1e-6)


def test_ignore_index_examples_are_excluded():
    logits, targets = random_logits_and_targets()
    kept = targets.clone()
    kept[::3] = -1
    with_ignored = ProbabilisticClassificationMetrics(5, 'logits', metrics=('NLL', 'Brier', 'n'))
    without = ProbabilisticClassificationMetrics(5, 'logits', metrics=('NLL', 'Brier', 'n'))
    with_ignored.update(NameDict(out=logits, target=kept))
    without.update(NameDict(out=logits[kept != -1], target=targets[kept != -1]))
    assert with_ignored.compute() == pytest.approx(without.compute())
    assert with_ignored.compute()['n'] == int((kept != -1).sum())


def test_per_class_means_and_the_class_balanced_mean():
    # Class 2 has no ground truth; class 0 is confidently right, class 1 confidently wrong.
    logits = torch.tensor([[5.0, 0.0, 0.0],
                           [5.0, 0.0, 0.0],
                           [5.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0, 1])
    m = ProbabilisticClassificationMetrics(3, 'logits',
                                           metrics=('cNLL', 'cBrier', 'mNLL', 'mBrier', 'NLL'))
    m.update(NameDict(out=logits, target=targets))
    r = m.compute()
    log_p = logits.double().log_softmax(1)
    assert r['cNLL'][0] == pytest.approx(-log_p[0, 0].item())
    assert r['cNLL'][1] == pytest.approx(-log_p[2, 1].item())
    assert math.isnan(r['cNLL'][2]) and math.isnan(r['cBrier'][2])
    # The balanced mean weights the two present classes equally, unlike the example mean.
    assert r['mNLL'] == pytest.approx((r['cNLL'][0] + r['cNLL'][1]) / 2)
    assert r['NLL'] == pytest.approx((2 * r['cNLL'][0] + r['cNLL'][1]) / 3)
    assert r['mNLL'] > r['NLL']


def test_zero_probability_on_the_true_class_is_an_infinite_nll():
    m = ProbabilisticClassificationMetrics(2, 'probs', metrics=('NLL', 'Brier'))
    m.update(NameDict(out=torch.tensor([[1.0, 0.0]]), target=torch.tensor([1])))
    r = m.compute()
    assert math.isinf(r['NLL'])
    assert r['Brier'] == pytest.approx(2.0)  # the maximum


def test_reset_clears_the_state():
    logits, targets = random_logits_and_targets()
    m = ProbabilisticClassificationMetrics(5, 'logits', metrics=('NLL', 'n'))
    m.update(NameDict(out=logits, target=targets))
    m.reset()
    assert m.compute()['n'] == 0
    assert math.isnan(m.compute()['NLL'])


def test_unknown_output_kind_and_wrong_shape_raise():
    with pytest.raises(ValueError, match="output_kind"):
        ProbabilisticClassificationMetrics(3, 'hard')
    m = ProbabilisticClassificationMetrics(3, 'logits')
    with pytest.raises(ValueError, match="shape"):
        m.update(NameDict(out=torch.zeros(4, 2), target=torch.zeros(4, dtype=torch.int64)))


# MCC and kappa from the confusion matrix ############################################

@pytest.mark.parametrize("seed", range(3))
def test_mcc_and_kappa_match_sklearn(seed):
    g = torch.Generator().manual_seed(seed)
    true = torch.randint(4, (300,), generator=g)
    pred = torch.where(torch.rand(300, generator=g) < 0.6, true,
                       torch.randint(4, (300,), generator=g))
    cm = multiclass_confusion_matrix(true, pred, 4)
    r = classification_metrics(cm, returns=('MCC', 'kappa'))
    true_np, pred_np = true.numpy(), pred.numpy()
    assert r['MCC'].item() == pytest.approx(matthews_corrcoef(true_np, pred_np), abs=1e-6)
    assert r['kappa'].item() == pytest.approx(cohen_kappa_score(true_np, pred_np), abs=1e-6)


def test_mcc_is_nan_not_zero_when_every_prediction_is_one_class():
    # sklearn returns 0 here, which reads as chance level. Kappa stays defined (and is 0):
    # its expected agreement is 0.7 < 1.
    cm = torch.tensor([[7, 0], [3, 0]])
    r = classification_metrics(cm, returns=('MCC', 'kappa'))
    assert math.isnan(r['MCC'].item())
    assert r['kappa'].item() == pytest.approx(0.0)


def test_both_are_nan_when_labels_and_predictions_are_one_class():
    cm = torch.tensor([[10, 0], [0, 0]])
    r = classification_metrics(cm, returns=('MCC', 'kappa'))
    assert math.isnan(r['MCC'].item()) and math.isnan(r['kappa'].item())


def test_mcc_and_kappa_through_classification_metrics_object():
    m = ClassificationMetrics(class_count=3, metrics=('MCC', 'kappa', 'A'))
    m.update(NameDict(out=torch.eye(3), target=torch.tensor([0, 1, 2])))
    r = m.compute()
    assert r['MCC'] == pytest.approx(1.0) and r['kappa'] == pytest.approx(1.0)
