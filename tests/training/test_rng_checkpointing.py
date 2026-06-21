"""Tests for reproducible-resume RNG checkpointing.

Covers the global RNG snapshot/restore utility and the trainer's `_RngCheckpoint`
adapter (global RNGs + the dedicated training-loader generator), which together
let a resumed run continue the random stream instead of restarting it.
"""
import random

import numpy as np
import torch

from vidlu.utils.rng import get_rng_state, set_rng_state
from vidlu.training.trainers import Trainer, _RngCheckpoint


def _global_draws():
    return (random.random(), tuple(np.random.rand(2).tolist()), torch.rand(2).tolist())


def test_global_rng_round_trip():
    state = get_rng_state()
    drawn = _global_draws()
    set_rng_state(state)
    assert _global_draws() == drawn


def test_set_rng_state_is_independent_of_intervening_draws():
    state = get_rng_state()
    expected = _global_draws()
    # Consume more entropy from every global source before restoring.
    random.random(); np.random.rand(5); torch.rand(5)
    set_rng_state(state)
    assert _global_draws() == expected


def test_rng_checkpoint_restores_globals_and_generator():
    gen = torch.Generator()
    gen.manual_seed(123)
    ckpt = _RngCheckpoint(gen)
    state = ckpt.state_dict()

    expected_global = _global_draws()
    expected_gen = torch.rand(3, generator=gen).tolist()

    ckpt.load_state_dict(state)
    assert _global_draws() == expected_global
    assert torch.rand(3, generator=gen).tolist() == expected_gen


def test_rng_checkpoint_state_is_torch_serializable(tmp_path):
    """The checkpoint state must round-trip through torch.save (plain data)."""
    gen = torch.Generator()
    gen.manual_seed(7)
    state = _RngCheckpoint(gen).state_dict()

    path = tmp_path / "rng.pt"
    torch.save(state, path)
    loaded = torch.load(path)

    gen2 = torch.Generator()
    _RngCheckpoint(gen2).load_state_dict(loaded)
    assert torch.equal(gen2.get_state(), gen.get_state())


def test_trainer_state_dict_attrs_includes_rng():
    assert "rng" in Trainer.state_dict_attrs
