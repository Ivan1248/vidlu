from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

from vidlu.training.extensions import TrainerExtension
from vidlu.data.dataset import Dataset
from .metrics import InternalMetricsProvider


def _get_underlying_dataset(dataset):
    """Get the underlying dataset if wrapped (e.g., by MapDataset)."""
    # MapDataset and other wrappers store the original dataset in .data
    while hasattr(dataset, 'data') and hasattr(dataset.data, '__getitem__'):
        dataset = dataset.data
    return dataset


def compute_attr_idx_to_class_occurrence_counts(dataset, class_counts: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """
    Compute class occurrence counts from dataset labels.
    
    This corresponds to `attribute_class_idx_to_occurrences` in the original IRAP GAIM code.
    Uses dict with GLOBAL attribute indices as keys to avoid index space confusion.
    
    Args:
        dataset: Dataset with segment_id_to_labels and segment_ids attributes.
        class_counts: Tuple of number of classes per attribute, e.g., (2, 3, 4).
    
    Returns:
        Dict mapping GLOBAL attr_idx to occurrence tensor. result[global_attr_idx][class_idx] = count.
    """
    counts = {global_attr_idx: torch.zeros(nc, dtype=torch.long) for global_attr_idx, nc in enumerate(class_counts)}
    
    # Check underlying dataset if wrapped (e.g., by MapDataset from prepare_dataset)
    underlying_ds = _get_underlying_dataset(dataset)
    
    # Require segment_id_to_labels (matches original implementation - no fallback)
    if not hasattr(underlying_ds, 'segment_id_to_labels'):
        raise AttributeError(
            f"Dataset (or its underlying dataset) must have 'segment_id_to_labels' attribute. "
            f"Got dataset type: {type(underlying_ds).__name__}. "
            f"This function requires a BihSequence dataset (or compatible dataset with segment_id_to_labels)."
        )
    if not hasattr(underlying_ds, 'segment_ids'):
        raise AttributeError(
            f"Dataset (or its underlying dataset) must have 'segment_ids' attribute. "
            f"Got dataset type: {type(underlying_ds).__name__}. "
            f"This function requires a BihSequence dataset (or compatible dataset with segment_ids)."
        )
    
    segment_ids = underlying_ds.segment_ids
    segment_id_to_labels = underlying_ds.segment_id_to_labels
    
    # Check for data consistency: all segment_ids should have labels
    missing_labels = [sid for sid in segment_ids if sid not in segment_id_to_labels]
    if missing_labels:
        raise ValueError(
            f"Found {len(missing_labels)} segment IDs without labels in segment_id_to_labels. "
            f"First few missing: {missing_labels[:5]}. "
            f"This indicates a data inconsistency - segments in segment_ids should have corresponding labels."
        )
    
    for sid in tqdm(segment_ids, desc="Computing class occurrence counts"):
        labels = segment_id_to_labels[sid]
        for global_attr_idx, nc in enumerate(class_counts):
            counts[global_attr_idx][int(labels[global_attr_idx])] += 1
    
    return counts


def add_attr_idx_to_class_occurrence_counts_to_info_lazily(ds: Dataset, cache_dir: str | Path, recompute: bool = False) -> Dataset:
    """
    Add class occurrence counts to dataset.info using HDD caching.
    
    This corresponds to `attribute_class_idx_to_occurrences` in the original IRAP GAIM code.
    The cached value is stored in dataset.info.attr_idx_to_class_occurrence_counts.
    
    Args:
        ds: Dataset with info.class_counts and underlying segment_id_to_labels/segment_ids
        cache_dir: Directory for caching (will create info_cache subdirectory)
        recompute: If True, force recomputation even if cache exists
        
    Returns:
        Dataset wrapped with HDDInfoCacheDataset that has info.attr_idx_to_class_occurrence_counts cached
        
    Example:
        >>> train_ds = make_bih_data()['train']
        >>> train_ds = add_attr_idx_to_class_occurrence_counts_to_info_lazily(train_ds, cache_dir="/path/to/cache")
        >>> # Now train_ds.info.attr_idx_to_class_occurrence_counts is available (computed lazily on first access)
    """
    def _compute(dataset: Dataset) -> dict[int, torch.Tensor]:
        """Compute class occurrence counts for caching."""
        if not hasattr(dataset, "info") or not hasattr(dataset.info, "class_counts"):
            raise ValueError(
                "Dataset must have info.class_counts for class occurrence count computation. "
                "Ensure the dataset was created with proper info attributes."
            )
        return compute_attr_idx_to_class_occurrence_counts(dataset, dataset.info.class_counts)
    
    return ds.info_cache_hdd(
        {"attr_idx_to_class_occurrence_counts": _compute},
        directory=cache_dir,
        recompute=recompute,
    )


def _estimate_random_classifier_recalls(n_classes: int) -> np.ndarray:
    return np.array([1.0 / n_classes] * n_classes, dtype=np.float64)


def _calculate_attr_idx_to_class_weights(
    attr_idx_to_class_occurrence_counts: dict[int, torch.Tensor],
    attr_idx_to_result_lists: dict[int, dict[str, np.ndarray]] | None = None,
    attr_idx_to_recalls: dict[int, np.ndarray] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Calculate class weights for dynamic balanced recall loss.
    
    This corresponds to `calculate_new_class_weights` in the original IRAP GAIM code.
    
    All arguments use GLOBAL attribute indices as dict keys to avoid index space confusion.
    
    Args:
        attr_idx_to_class_occurrence_counts: Dict mapping GLOBAL attribute index to occurrence tensor.
            Corresponds to `attribute_class_idx_to_occurrences` in original code.
            attr_idx_to_class_occurrence_counts[global_attr_idx][class_idx] = count.
        attr_idx_to_result_lists: Optional dict mapping GLOBAL attribute index to result dicts
            with 'y_true' and 'y_pred' keys. If None and attr_idx_to_recalls is None,
            uses random classifier recalls.
        attr_idx_to_recalls: Optional dict mapping GLOBAL attribute index to recall arrays.
            If provided, used directly instead of computing from result_lists.
    
    Returns:
        Dict mapping GLOBAL attribute index to weight tensor (attr_idx_to_class_weights).
    """
    # Build recalls for all attributes that have occurrence counts
    attr_idx_to_class_recalls = dict()
    if attr_idx_to_recalls is not None:
        attr_idx_to_class_recalls = attr_idx_to_recalls
    elif attr_idx_to_result_lists is None:
        # No results yet - estimate random classifier recalls for all attributes
        for global_attr_idx, occ_counts in attr_idx_to_class_occurrence_counts.items():
            nc = len(occ_counts)
            attr_idx_to_class_recalls[global_attr_idx] = _estimate_random_classifier_recalls(nc)
    else:
        # Compute recalls only for attributes that have results
        for global_attr_idx, result_lists in attr_idx_to_result_lists.items():
            nc = len(attr_idx_to_class_occurrence_counts[global_attr_idx])
            labels = list(range(nc))
            attr_idx_to_class_recalls[global_attr_idx] = recall_score(
                result_lists['y_true'], result_lists['y_pred'], average=None, labels=labels, zero_division=1
            )

    # Compute weights only for attributes that have both occurrences AND recalls
    attr_idx_to_class_weights = {}
    for global_attr_idx, occ_counts in attr_idx_to_class_occurrence_counts.items():
        if global_attr_idx not in attr_idx_to_class_recalls:
            continue  # Skip attributes without recall data
        total = occ_counts.sum().item()
        occ = np.array([int(occ_counts[c].item()) for c in range(len(occ_counts))], dtype=np.float64)
        zero_occ_const = 1e-4
        inv_freq = np.array([(total / c) if c > 0 else zero_occ_const for c in occ], dtype=np.float64)
        class_recalls = np.asarray(attr_idx_to_class_recalls[global_attr_idx], dtype=np.float64)
        w = inv_freq * (1.0 - class_recalls) + np.sqrt(inv_freq) * class_recalls
        attr_idx_to_class_weights[global_attr_idx] = torch.tensor(w, dtype=torch.float32)
    return attr_idx_to_class_weights


def _find_dataset_by_prefix(data: dict[str, Dataset] | None, prefix: str) -> Dataset | None:
    """Find dataset in data dict by key prefix."""
    if data is None:
        return None
    return next((ds for name, ds in data.items() if name.startswith(prefix)), None)


def _get_attr_idx_to_class_occurrence_counts(
    dataset: Dataset, 
    cache_dir: Path,
) -> dict[int, torch.Tensor]:
    """
    Get class occurrence counts from dataset, using cache if available or computing if needed.
    
    This corresponds to `attribute_class_idx_to_occurrences` in the original IRAP GAIM code.
    
    Args:
        dataset: Dataset with info.class_counts and underlying segment_id_to_labels/segment_ids
        cache_dir: Cache directory for caching (will cache on-the-fly if not already cached)
        
    Returns:
        Dict mapping GLOBAL attr_idx to occurrence count tensor
    """
    try:
        return dataset.info.attr_idx_to_class_occurrence_counts
    except AttributeError:
        dataset = add_attr_idx_to_class_occurrence_counts_to_info_lazily(dataset, cache_dir, recompute=False)
        return dataset.info.attr_idx_to_class_occurrence_counts


def _determine_attrs_idx(
    dataset: Dataset,
    attrs_idx: Sequence[int] | None,
) -> list[int]:
    """Determine attribute indices to use, either from explicit parameter or dataset info."""
    if attrs_idx is not None:
        return list(attrs_idx)
    
    # Automatically use attrs_to_include from dataset
    if not hasattr(dataset, 'info') or not hasattr(dataset.info, 'attribute_names'):
        raise ValueError(
            "DynamicBalancedRecallWeights requires dataset.info.attribute_names to determine attribute subset. "
            "Either provide attrs_idx explicitly when creating DynamicBalancedRecallWeights, "
            "or ensure the dataset has info.attribute_names set (e.g., by using make_bih_data())."
        )
    from .attrs import get_attrs_to_include, map_attr_names_to_indices
    attr_names = get_attrs_to_include()
    return map_attr_names_to_indices(attr_names, dataset.info.attribute_names)


class DynamicBalancedRecallWeights(TrainerExtension):
    """
    After each epoch, recompute per-attribute class weights from validation results (macro recalls)
    and swap the trainer.loss with a wrapper that injects weights into cross-entropy per attribute.
    
    Class occurrence counts (corresponding to `attribute_class_idx_to_occurrences` in the original
    IRAP GAIM code) are cached using dataset.info.attr_idx_to_class_occurrence_counts if available,
    or cached on-the-fly using the provided cache_dir.
    
    Args:
        cache_dir: Cache directory for storing class occurrence counts (required)
        dataset_split_prefix: Prefix for validation dataset key (default: 'val').
            Ignored if split_names is provided.
        split_names: Explicit list of split names to use for weight updates (default: None).
            If None, auto-detect by prefix. If provided, dataset_split_prefix is ignored.
        attrs_idx: Explicit attribute indices to use, or None to auto-detect from dataset
        
    Note: 
        - Requires an InternalMetricsProvider metric (e.g., MultiAttributeClassificationMetrics) 
          to be included in trainer.metrics.
        - For best performance, pre-cache using add_attr_idx_to_class_occurrence_counts_to_info_lazily
          in your data factory. The cache_dir is still required for on-the-fly caching fallback.
    """

    def __init__(
        self, 
        cache_dir: str | Path,
        dataset_split_prefix: str = 'val',
        split_names: Sequence[str] | None = None,
        attrs_idx: Sequence[int] | None = None,
    ):
        self.dataset_split_prefix = dataset_split_prefix
        self.split_names = list(split_names) if split_names is not None else None
        self.attrs_idx = attrs_idx
        self.cache_dir = Path(cache_dir)
        self.attr_idx_to_class_occurrence_counts: dict[int, torch.Tensor] | None = None
        self.attr_idx_to_class_weights: dict[int, torch.Tensor] | None = None
        self.loss_adapter = None

    def initialize(self, trainer):
        trainer.model.eval()
        
        if trainer.data is None:
            raise ValueError("DynamicBalancedRecallWeights requires trainer.data to be set.")
        
        # Find and validate training dataset
        train_ds = _find_dataset_by_prefix(trainer.data, 'train')
        if train_ds is None:
            raise ValueError("No training dataset found in trainer.data (expected key starting with 'train').")

        # Get class occurrence counts (using cache if available)
        self.attr_idx_to_class_occurrence_counts = _get_attr_idx_to_class_occurrence_counts(train_ds, self.cache_dir)
        
        # Determine which attributes to include
        self.attrs_idx = _determine_attrs_idx(train_ds, self.attrs_idx)

        # Ensure loss can receive dynamic weights and attribute filtering
        self.loss_adapter = self._bind_loss_adapter(trainer.loss)
        self.loss_adapter.set_attrs_idx(self.attrs_idx)

        # Initialize weights from class occurrence counts (no result_lists yet, uses random classifier recalls)
        self.attr_idx_to_class_weights = _calculate_attr_idx_to_class_weights(self.attr_idx_to_class_occurrence_counts)
        self.loss_adapter.set_class_weights(self.attr_idx_to_class_weights)

        # Determine target validation split names
        if self.split_names is None:
            # Auto-detect by prefix
            self.target_split_names = [
                name for name in trainer.data.keys() 
                if name.startswith(self.dataset_split_prefix)
            ]
        else:
            # Use explicit split names
            self.target_split_names = [
                name for name in self.split_names 
                if name in trainer.data
            ]
            missing = set(self.split_names) - set(self.target_split_names)
            if missing:
                raise ValueError(
                    f"DynamicBalancedRecallWeights: Split names {missing} not found in trainer.data. "
                    f"Available splits: {list(trainer.data.keys())}"
                )

        # Store trainer reference for lazy metric lookup
        self._trainer = trainer
        self._metric = None

        # Register epoch end handler on evaluation loop
        @trainer.evaluation.epoch_completed.handler
        def on_eval_epoch_end(state):
            split_name = getattr(state, "split_name", "")
            # Check if this split should trigger weight update
            if self.split_names is not None:
                # Explicit split names: exact match
                if split_name in self.target_split_names:
                    self._update_weights_from_reused_eval(trainer, state)
            else:
                # Prefix-based: check if split starts with prefix
                if split_name.startswith(self.dataset_split_prefix):
                    self._update_weights_from_reused_eval(trainer, state)

    def _get_metric(self):
        """Lazily find the InternalMetricsProvider metric from trainer.metrics."""
        if self._metric is None:
            self._metric = next(
                (m for m in self._trainer.metrics if isinstance(m, InternalMetricsProvider)),
                None
            )
            if self._metric is None:
                raise RuntimeError(
                    "DynamicBalancedRecallWeights: No InternalMetricsProvider metric found in trainer.metrics. "
                    "Ensure MultiAttributeClassificationMetrics is included in metrics."
                )
        return self._metric

    def _update_weights_from_reused_eval(self, trainer, state):
        """Update weights using stats from the just-completed evaluation."""
        # Get internal metrics directly from metric (always available via get_internal_metrics())
        metric = self._get_metric()
        stats = metric.get_internal_metrics()
        
        if not stats:
            raise RuntimeError(
                "DynamicBalancedRecallWeights: Metric returned empty stats. "
                "Ensure the metric has been updated with evaluation data."
            )
        
        # Compute recalls from stats
        attr_idx_to_recalls = {}
        for attr_idx, s in stats.items():
            tp = s['tp'].numpy()
            actual = s['actual'].numpy()
            # Handle zero division: if actual is 0, recall is 1 (matches zero_division=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = tp / actual
                r[actual == 0] = 1.0
            attr_idx_to_recalls[attr_idx] = r
        
        self.attr_idx_to_class_weights = _calculate_attr_idx_to_class_weights(
            self.attr_idx_to_class_occurrence_counts, attr_idx_to_recalls=attr_idx_to_recalls
        )
        self.loss_adapter.set_class_weights(self.attr_idx_to_class_weights)

    def _bind_loss_adapter(self, loss_callable):
        required_methods = ("set_attrs_idx", "set_class_weights")
        for method in required_methods:
            if not hasattr(loss_callable, method):
                raise TypeError(
                    "DynamicBalancedRecallWeights requires the configured loss to support "
                    f"{method}(). Use MultiAttributeCrossEntropyLoss or supply a compatible loss wrapper."
                )
            if not callable(getattr(loss_callable, method)):
                raise TypeError(
                    f"DynamicBalancedRecallWeights expected trainer.loss.{method} to be callable."
                )
        return loss_callable


