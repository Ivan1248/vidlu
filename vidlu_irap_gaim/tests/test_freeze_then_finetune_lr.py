import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiplicativeLR

from vidlu_irap_gaim.training.extensions import FreezeThenFinetune, phase_lr_decay_factor


def test_factor_equals_original_at_original_phase_lengths():
    assert phase_lr_decay_factor(0.8 ** 2, 2) == pytest.approx(0.8)
    assert phase_lr_decay_factor(0.88 ** 13, 13) == pytest.approx(0.88)


@pytest.mark.parametrize("num_epochs", [1, 2, 8, 13])
def test_total_decay_invariant_to_phase_length(num_epochs):
    for total in (0.8 ** 2, 0.88 ** 13):
        assert phase_lr_decay_factor(total, num_epochs) ** num_epochs == pytest.approx(total)


def test_multiplicative_lr_reaches_total_decay():
    base_lr, total, num_epochs = 1e-5, 0.88 ** 13, 8
    opt = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=base_lr)
    factor = phase_lr_decay_factor(total, num_epochs)
    scheduler = MultiplicativeLR(opt, lr_lambda=lambda epoch: factor)
    for _ in range(num_epochs):
        scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * total)


class _RecordingEvent:
    def __init__(self):
        self.handlers = []

    def handler(self, f):
        self.handlers.append(f)
        return f


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def get_trainable_parameters(self):
        return list(self.head.parameters())


class _FakeTrainer:
    def __init__(self, epoch_count):
        self.model = _TinyModel()
        self.epoch_count = epoch_count
        self.training = SimpleNamespace(epoch_started=_RecordingEvent())

    def load_state_dict(self, state_dict):
        pass


def _current_lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("epoch_count,frozen_factor,finetune_factor",
                         [(15, 0.8, 0.88),
                          (10, 0.8, 0.88 ** (13 / 8))])
def test_extension_installs_adaptive_schedulers(epoch_count, frozen_factor, finetune_factor):
    trainer = _FakeTrainer(epoch_count=epoch_count)
    FreezeThenFinetune(num_frozen_epochs=2).initialize(trainer)

    assert _current_lr(trainer) == pytest.approx(5e-5)
    trainer.lr_scheduler.step()
    assert _current_lr(trainer) == pytest.approx(5e-5 * frozen_factor)

    (on_epoch_started,) = trainer.training.epoch_started.handlers
    on_epoch_started(SimpleNamespace(epoch=2))

    assert _current_lr(trainer) == pytest.approx(1e-5)
    trainer.lr_scheduler.step()
    assert _current_lr(trainer) == pytest.approx(1e-5 * finetune_factor)


def test_extension_rejects_epoch_count_without_finetune_epochs():
    trainer = _FakeTrainer(epoch_count=2)
    with pytest.raises(ValueError, match="epoch_count"):
        FreezeThenFinetune(num_frozen_epochs=2).initialize(trainer)
