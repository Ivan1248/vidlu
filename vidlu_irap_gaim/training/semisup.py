"""Semi-supervised training support for vidlu_irap_gaim.

This module provides components to enable semi-supervised learning with the
BIH dataset, leveraging vidlu's existing SemisupConsStep infrastructure.
"""

import typing as T
from functools import partial
from pathlib import Path
import random

import torch
import numpy as np

from vidlu.modules.losses import kl_div_ll

from vidlu_irap_gaim.data.bih_dataset import make_bih_data


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
    labeled_ratio: float | None = None,
    labeled_size: int | None = None,
    use_all_as_unlabeled: bool = False,
    seed: int = 42,
    shuffle: bool = True,
    *,
    metadata_dir: T.Union[str, Path] | None = None,
    **kwargs,
) -> dict:
    """Create semi-supervised BiH datasets with labeled/unlabeled train split.

    Uses a synthetic split: a portion of labeled training data is treated as
    "labeled" while the rest is treated as "unlabeled" (labels ignored).

    Args:
        labeled_ratio: Fraction of training data to use as labeled (0.0-1.0).
        use_all_as_unlabeled: If True, use the full training set as unlabeled
            in addition to the labeled split (same dataset for both roles).
        seed: Random seed for reproducible shuffling (only used if shuffle=True).
        shuffle: If True (default), shuffle training data before splitting.
            If False, take the first labeled_ratio fraction without shuffling.
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
    ds = make_bih_data(metadata_dir=metadata_dir, **kwargs)
    train_ds = ds["train"]


    if labeled_ratio is None and labeled_size is None:
        raise ValueError("Either labeled_ratio or labeled_size must be provided.")
    if labeled_size is not None:
        labeled_ratio = labeled_size / len(train_ds)
        print(f"Using labeled_size={labeled_size} -> labeled_ratio={labeled_ratio}")
    if labeled_ratio is not None:
        labeled_size = int(labeled_ratio * len(train_ds) + 0.5)
        print(f"Using labeled_ratio={labeled_ratio} -> labeled_size={labeled_size}")

    if shuffle:
        train_ds = train_ds.permute(seed=seed)
    if use_all_as_unlabeled:
        train_l, _ = train_ds.split(ratio=labeled_ratio)
        train_u = ds["train"]
    else:
        train_l, train_u = train_ds.split(ratio=labeled_ratio)
    return {"train": train_l, "train_u": train_u, "val": ds["val"], "test": ds["test"]}


def get_hard_pseudo_labels(
    logits_tuple: tuple[torch.Tensor, ...],
    temperature: float = 1.0,
    conf_thresh: T.Union[float, dict[int, float]] = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert multi-attribute logit tuple to hard pseudo-labels with confidence masking.

    Args:
        logits_tuple: Tuple of (B, K_i) logit tensors, length A (num attributes).
        temperature: Temperature for softmax scaling. >1 flattens, <1 sharpens confidence.
        conf_thresh: Confidence threshold(s). Can be:
            - float: Single global threshold applied to all attributes (fixed mode)
            - dict: Per-attribute thresholds, mapping attr_idx -> float (adaptive mode)

    Returns:
        labels: (B, A) int64 tensor, -1 where max_prob <= threshold for that attribute
        mask:   (B, A) bool tensor, True where label is valid
    """
    B = logits_tuple[0].shape[0]
    device = logits_tuple[0].device
    A = len(logits_tuple)
    labels = torch.full((B, A), -1, dtype=torch.long, device=device)
    mask = torch.zeros((B, A), dtype=torch.bool, device=device)

    for i, logits in enumerate(logits_tuple):
        scaled = logits / temperature if temperature != 1.0 else logits
        probs = torch.softmax(scaled, dim=1)
        max_prob, argmax = probs.max(dim=1)

        # Get threshold for this attribute (adaptive or fixed)
        thresh = conf_thresh[i] if isinstance(conf_thresh, dict) else conf_thresh

        valid = max_prob > thresh
        labels[:, i] = torch.where(valid, argmax, torch.full_like(argmax, -1))
        mask[:, i] = valid

    return labels, mask


def update_adaptive_thresholds(
    logits_tuple: tuple[torch.Tensor, ...],
    thresholds: dict[int, float],
    ema_momentum: float = 0.999,
    aggregation_fn: T.Callable[[torch.Tensor], float] | None = None,
) -> dict[int, float]:
    """Update per-attribute confidence thresholds using exponential moving average (MC-PanDA++ style).

    This allows threshold tau_a to adapt to the evolving confidence distribution of each attribute,
    automatically handling class imbalance within attributes. Each attribute may have different
    dominant/rare class distributions.

    Args:
        logits_tuple: Tuple of (B, K_i) logit tensors from teacher on clean unlabeled data.
        thresholds: Current per-attribute thresholds dict, mapping attr_idx -> float.
        ema_momentum: EMA momentum (alpha in Eq. 5 of MC-PanDA++). Higher = slower updates.
        aggregation_fn: Function to aggregate confidence scores per attribute (default: 75th percentile).
                       Called as aggregation_fn(confidences_array) -> scalar.

    Returns:
        Updated thresholds dict.
    """
    if aggregation_fn is None:
        aggregation_fn = lambda arr: torch.quantile(arr, 0.75).item()

    updated = {}
    for i, logits in enumerate(logits_tuple):
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1).values  # (B,)

        # Aggregate confidence for this attribute (e.g., 75th percentile)
        delta = aggregation_fn(max_probs)

        # EMA update: tau_a^n = alpha * tau_a^(n-1) + (1 - alpha) * delta_a^n
        old_thresh = thresholds.get(i, 0.0)
        new_thresh = ema_momentum * old_thresh + (1 - ema_momentum) * delta
        updated[i] = new_thresh

    return updated


class PseudoLabeledDataset:
    """Wraps a dataset and replaces its `target` field with stored pseudo-labels.

    Args:
        dataset: Underlying unlabeled dataset (BihSequence or similar).
        pseudo_labels: Path to .npz file (with 'labels' array of shape (N, A),
                       int16, -1 for masked) or a (N, A) numpy array directly.

    The .npz is produced by tools/generate_pseudo_labels.py.
    """

    def __init__(self, dataset, pseudo_labels: T.Union[str, Path, np.ndarray]):
        self._dataset = dataset
        if isinstance(pseudo_labels, (str, Path)):
            npz = np.load(pseudo_labels, allow_pickle=False)
            arr = npz['labels']
            if len(arr) != len(dataset):
                raise ValueError(f"Pseudo-label count {len(arr)} != dataset size {len(dataset)}")
            self._labels = arr  # (N, A) int16
        else:
            self._labels = np.asarray(pseudo_labels)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        from vidlu.data import Record
        record = self._dataset[idx]
        pseudo_target = torch.from_numpy(self._labels[idx].astype(np.int64))
        return Record(record, target=pseudo_target)
