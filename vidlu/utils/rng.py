"""Snapshot/restore of global RNG state for reproducible checkpoint resume.

These capture the process-global RNGs (Python ``random``, NumPy, torch CPU, and
torch CUDA per device) as plain, serializable data so a resumed run can continue
the random stream instead of restarting it. A dedicated training-loader generator
is checkpointed separately by the trainer (see ``vidlu.training.trainers``).
"""
import random
import warnings

import numpy as np
import torch


def get_rng_state() -> dict:
    """Snapshot the process-global RNGs as plain serializable data."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()  # one tensor per device
    return state


def to_cpu(x):
    """Recursively move all PyTorch tensors in a nested structure to CPU."""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def set_rng_state(state: dict) -> None:
    """Restore the process-global RNGs from a snapshot produced by `get_rng_state`."""
    state = to_cpu(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        else:
            warnings.warn(
                f"Saved CUDA RNG state has {len(cuda)} device(s) but "
                f"{torch.cuda.device_count()} are visible; skipping CUDA RNG restore.")

