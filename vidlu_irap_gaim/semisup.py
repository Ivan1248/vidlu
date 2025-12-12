"""Semi-supervised training support for vidlu_irap_gaim.

This module provides components to enable semi-supervised learning with the
BIH dataset, leveraging vidlu's existing SemisupConsStep infrastructure.
"""

from __future__ import annotations

import typing as T
from functools import partial
from pathlib import Path
import random

import torch
import numpy as np

from vidlu.modules.losses import kl_div_ll

from .datasets import make_bih_data


def multi_attribute_kl_div_ll(
    logits_p: tuple[torch.Tensor, ...], logits_t: tuple[torch.Tensor, ...], reduction: str = "mean"
) -> torch.Tensor:
    """KL divergence across multi-attribute tuple outputs (logits to logits).

    Args:
        logits_p: Tuple of predicted logits tensors, one per attribute.
        logits_t: Tuple of target logits tensors (teacher/pseudo-labels).
        reduction: "mean" or "none". If "none", returns per-sample losses.

    Returns:
        If reduction="mean": scalar loss (mean over samples and attributes).
        If reduction="none": tensor of shape (B,) with per-sample mean loss.
    """
    if len(logits_p) != len(logits_t):
        raise ValueError(f"Number of attributes mismatch: {len(logits_p)} vs {len(logits_t)}")

    # Compute KL divergence for each attribute
    losses = [kl_div_ll(p, t) for p, t in zip(logits_p, logits_t)]

    # Stack: shape (n_attrs, B)
    stacked = torch.stack(losses)

    if reduction == "none":
        # Return per-sample mean across attributes: shape (B,)
        return stacked.mean(0)
    else:
        # Return scalar: mean over both samples and attributes
        return stacked.mean()


def make_semisup_bih_data(
    labeled_ratio: float = 0.1,
    use_all_as_unlabeled: bool = False,
    seed: int = 42,
    *,
    metadata_dir: T.Union[str, Path] | None = None,
    **bih_kwargs,
) -> dict:
    """Create semi-supervised BiH datasets with labeled/unlabeled train split.

    Uses a synthetic split: a portion of labeled training data is treated as
    "labeled" while the rest is treated as "unlabeled" (labels ignored).

    Args:
        labeled_ratio: Fraction of training data to use as labeled (0.0-1.0).
        use_all_as_unlabeled: If True, use the full training set as unlabeled
            in addition to the labeled split (same dataset for both roles).
        seed: Random seed for reproducible splitting.
        metadata_dir: Optional override for IRAP_BIH_METADATA directory.
        **bih_kwargs: Additional arguments passed to make_bih_data() (e.g. irap_home).

    Returns:
        Dict with keys:
        - 'train': labeled dataset
        - 'train_u': unlabeled dataset
        - 'val': validation dataset
        - 'test': test dataset

    Example:
        >>> data = make_semisup_bih_data(labeled_ratio=0.1)
        >>> train_labeled = data['train']
        >>> train_unlabeled = data['train_u']
    """
    ds = make_bih_data(metadata_dir=metadata_dir, **bih_kwargs)
    if use_all_as_unlabeled:
        train_l = ds["train"]
        train_u = ds["train"]
    else:
        train_l, train_u = ds["train"].permute(seed=seed).split(ratio=labeled_ratio)
    return {"train": train_l, "train_u": train_u, "val": ds["val"], "test": ds["test"]}
