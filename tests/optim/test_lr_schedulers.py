"""Tests for the warmup shape and the multiplicative warmup-cosine schedule.

`WarmupCosineLR` exists to serve layer-wise LR decay, whose defining property is the ratio
between the learning rates of different parameter groups. These check that the schedule does
not flatten it.
"""

import pytest
import torch
from torch import nn

from vidlu.optim import lr_shapes
from vidlu.optim.lr_schedulers import CosineLR, ScalableLR, WarmupCosineLR


def _make_optimizer(lrs):
    return torch.optim.AdamW([dict(params=[nn.Parameter(torch.zeros(1))], lr=lr) for lr in lrs])


def test_each_group_gets_its_own_shape():
    optimizer = _make_optimizer([1., 1.])
    ScalableLR(optimizer, func=[lambda p: 0.25, lambda p: 0.75], epoch_count=10)
    assert [g["lr"] for g in optimizer.param_groups] == [0.25, 0.75]


def test_stepping_past_the_epoch_count_is_rejected():
    """The shapes are only defined on `[0, 1]`. Outside it they fail quietly: `ramp` turns
    negative, `poly` returns a complex number, and `cosine_lr` turns back upwards."""
    optimizer = _make_optimizer([1e-4])
    scheduler = ScalableLR(optimizer, func=lr_shapes.ramp, epoch_count=2)
    scheduler.step()
    scheduler.step()
    with pytest.raises(ValueError, match="past epoch_count"):
        scheduler.step()


def test_with_warmup_endpoints():
    shape = lr_shapes.with_warmup(lr_shapes.cosine_lr, warmup_proportion=0.1, start_factor=0.1,
                                  min_factor=0.01)
    # The factor at progress 0 is positive: schedulers step per epoch, so a zero-valued first
    # point would waste a whole epoch.
    assert shape(0.) == pytest.approx(0.1)
    assert shape(0.1) == pytest.approx(1.)
    assert shape(1.) == pytest.approx(0.01)


def test_with_warmup_rescales_the_wrapped_shape():
    shape = lr_shapes.with_warmup(lr_shapes.ramp, warmup_proportion=0.5, start_factor=0.,
                                  min_factor=0.)
    assert shape(0.25) == pytest.approx(0.5)  # halfway through the warmup
    assert shape(0.75) == pytest.approx(0.5)  # halfway through the decay


def test_warmup_cosine_preserves_layer_wise_ratios():
    optimizer = _make_optimizer([1e-4 * 0.65 ** i for i in range(5)])
    scheduler = WarmupCosineLR(optimizer, epoch_count=10, warmup_proportion=0.1, min_factor=0.01)
    for _ in range(10):
        scheduler.step()
        lrs = [g["lr"] for g in optimizer.param_groups]
        assert all(b / a == pytest.approx(0.65) for a, b in zip(lrs, lrs[1:]))


def test_absolute_eta_min_would_flatten_the_decay():
    """The reason `WarmupCosineLR` exists: `CosineLR`'s `eta_min` is an absolute floor applied
    identically to every group, so a layer-wise ladder collapses onto it, and the groups whose
    rates were decayed the most end up raised instead."""
    lrs = [1e-4 * 0.65 ** i for i in range(12)]
    optimizer = _make_optimizer(lrs)
    scheduler = CosineLR(optimizer, epoch_count=10, eta_min=1e-6)
    for _ in range(10):
        scheduler.step()
    final = [g["lr"] for g in optimizer.param_groups]
    assert all(lr == pytest.approx(1e-6) for lr in final)
    assert final[-1] > lrs[-1]
