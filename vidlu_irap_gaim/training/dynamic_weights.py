from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

from vidlu.training.extensions import TrainerExtension
from irap_data.dataset import Dataset
from vidlu_irap_gaim.metrics import InternalMetricsProvider


def compute_attr_idx_to_class_occurrence_counts(dataset, class_counts: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Compatibility wrapper for legacy index-based calls."""
    attr_to_class_count = {i: c for i, c in enumerate(class_counts)}
    return compute_attr_key_to_class_occurrence_counts(dataset, attr_to_class_count)


def compute_attr_key_to_class_occurrence_counts(
    dataset, attr_to_class_count: dict[object, int], attr_to_index: dict[object, int] | None = None
) -> dict[object, torch.Tensor]:
    """
    Compute class occurrence counts from dataset labels.

    This corresponds to `attribute_class_idx_to_occurrences` in the original IRAP GAIM code.
    Uses dict with Generic Keys (Names or Indices) to match metrics configuration.

    Args:
        dataset: Dataset whose `info` has `segment_id_to_labels` and `segment_ids`.
        attr_to_class_count: Dict mapping key -> number of classes.
        attr_to_index: Dict mapping key -> global attribute index. If None, assumes keys are indices.

    Returns:
        Dict mapping Key to occurrence tensor. result[key][class_idx] = count.
    """
    counts = {key: torch.zeros(nc, dtype=torch.long) for key, nc in attr_to_class_count.items()}

    for required_attr in ("segment_id_to_labels", "segment_ids"):
        if not hasattr(dataset.info, required_attr):
            raise AttributeError(
                f"dataset.info must have the '{required_attr}' attribute, which an IRAPDataset"
                f" provides. Got a dataset of type {type(dataset).__name__}.")

    segment_ids = dataset.info.segment_ids
    segment_id_to_labels = dataset.info.segment_id_to_labels

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
        for key in counts.keys():
            # Determine global index
            if attr_to_index is not None:
                idx = attr_to_index[key]
            else:
                idx = key if isinstance(key, int) else -1

            if idx >= 0 and idx < len(labels):
                counts[key][int(labels[idx])] += 1

    return counts


def add_attr_idx_to_class_occurrence_counts_to_info_lazily(
    ds: Dataset, cache_dir: str | Path, recompute: bool = False
) -> Dataset:
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

    breakpoint() # don't forget that caching is happening

    return ds.info_cache_hdd(
        {"attr_idx_to_class_occurrence_counts": _compute},
        directory=cache_dir,
        recompute=recompute,
    )


def _estimate_random_classifier_recalls(n_classes: int) -> np.ndarray:
    return np.array([1.0 / n_classes] * n_classes, dtype=np.float64)


def _calculate_attr_key_to_class_weights(
    attr_key_to_class_occurrence_counts: dict[object, torch.Tensor],
    attr_key_to_result_lists: dict[object, dict[str, np.ndarray]] | None = None,
    attr_key_to_recalls: dict[object, np.ndarray] | None = None,
) -> dict[object, torch.Tensor]:
    """
    Calculate class weights for dynamic balanced recall loss.

    This corresponds to `calculate_new_class_weights` in the original IRAP GAIM code.

    All arguments use Generic Keys (likely Strings) as dict keys.

    Args:
        attr_key_to_class_occurrence_counts: Dict mapping Key to occurrence tensor.
            Corresponds to `attribute_class_idx_to_occurrences` in original code.
            attr_key_to_class_occurrence_counts[key][class_idx] = count.
        attr_key_to_result_lists: Optional dict mapping Key to result dicts
            with 'y_true' and 'y_pred' keys. If None and attr_key_to_recalls is None,
            uses random classifier recalls.
        attr_key_to_recalls: Optional dict mapping Key to recall arrays.
            If provided, used directly instead of computing from result_lists.

    Returns:
        Dict mapping Key to weight tensor (attr_key_to_class_weights).
    """
    # Build recalls for all attributes that have occurrence counts
    attr_key_to_class_recalls = dict()
    if attr_key_to_recalls is not None:
        attr_key_to_class_recalls = attr_key_to_recalls
    elif attr_key_to_result_lists is None:
        # No results yet - estimate random classifier recalls for all attributes
        for key, occ_counts in attr_key_to_class_occurrence_counts.items():
            nc = len(occ_counts)
            attr_key_to_class_recalls[key] = _estimate_random_classifier_recalls(nc)
    else:
        # Compute recalls only for attributes that have results
        for key, result_lists in attr_key_to_result_lists.items():
            nc = len(attr_key_to_class_occurrence_counts[key])
            labels = list(range(nc))
            attr_key_to_class_recalls[key] = recall_score(
                result_lists["y_true"], result_lists["y_pred"], average=None, labels=labels, zero_division=1
            )

    # Compute weights only for attributes that have both occurrences AND recalls
    attr_key_to_class_weights = {}
    for key, occ_counts in attr_key_to_class_occurrence_counts.items():
        if key not in attr_key_to_class_recalls:
            continue  # Skip attributes without recall data
        total = occ_counts.sum().item()
        occ = np.array([int(occ_counts[c].item()) for c in range(len(occ_counts))], dtype=np.float64)
        zero_occ_const = 1e-4
        inv_freq = np.array([(total / c) if c > 0 else zero_occ_const for c in occ], dtype=np.float64)
        class_recalls = np.asarray(attr_key_to_class_recalls[key], dtype=np.float64)
        w = inv_freq * (1.0 - class_recalls) + np.sqrt(inv_freq) * class_recalls
        attr_key_to_class_weights[key] = torch.tensor(w, dtype=torch.float32)
    return attr_key_to_class_weights


def _find_dataset_by_prefix(data: dict[str, Dataset] | None, prefix: str) -> Dataset | None:
    """Find dataset in data dict by key prefix."""
    if data is None:
        return None
    return next((ds for name, ds in data.items() if name.startswith(prefix)), None)


def _get_attr_key_to_class_occurrence_counts(
    dataset: Dataset,
    cache_dir: Path,
    attr_to_class_count: dict[object, int],
    attr_to_index: dict[object, int] | None = None,
) -> dict[object, torch.Tensor]:
    try:
        # Try new cache name
        return dataset.info.attr_key_to_class_occurrence_counts
    except AttributeError:
        # Helper to attach cache lazily
        def _compute(ds):
            return compute_attr_key_to_class_occurrence_counts(ds, attr_to_class_count, attr_to_index)

        # We rename the property on info to attr_key_...
        dataset = dataset.info_cache_hdd(
            {"attr_key_to_class_occurrence_counts": _compute},
            directory=cache_dir,
            recompute=False,
        )
        return dataset.info.attr_key_to_class_occurrence_counts


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
        attrs_to_include: Explicit attribute names to use, or None to auto-detect from dataset

    Note:
        - Requires an InternalMetricsProvider metric (e.g., MultiAttributeClassificationMetrics)
          to be included in trainer.metrics.
        - For best performance, pre-cache using add_attr_idx_to_class_occurrence_counts_to_info_lazily
          in your data factory. The cache_dir is still required for on-the-fly caching fallback.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        dataset_split_prefix: str = "val",
        split_names: Sequence[str] | None = None,
        attrs_to_include: Sequence[str] | None = None,
    ):
        self.dataset_split_prefix = dataset_split_prefix
        self.split_names = list(split_names) if split_names is not None else None
        self.attrs_to_include = attrs_to_include
        self.cache_dir = Path(cache_dir)
        self.attr_key_to_class_occurrence_counts: dict[object, torch.Tensor] | None = None
        self.attr_key_to_class_weights: dict[object, torch.Tensor] | None = None
        self.loss_adapter = None
        self.attr_to_index = None

    def initialize(self, trainer):
        trainer.model.eval()

        if trainer.data is None:
            raise ValueError("DynamicBalancedRecallWeights requires trainer.data to be set.")

        # Find and validate training dataset
        train_ds = _find_dataset_by_prefix(trainer.data, "train")
        if train_ds is None:
            raise ValueError("No training dataset found in trainer.data (expected key starting with 'train').")

        # Determine which attributes to include and construct mappings
        # If attrs_to_include is None, we default to ALL attributes from dataset if possible
        if self.attrs_to_include is None:
            if hasattr(train_ds, "info") and hasattr(train_ds.info, "attribute_names"):
                self.attrs_to_include = tuple(train_ds.info.attribute_names)

        # Construct Key -> Index mapping
        if self.attrs_to_include is not None:
            from irap_data.attrs import map_attr_names_to_indices

            attrs_idx_list = map_attr_names_to_indices(
                self.attrs_to_include, train_ds.info.attr_to_value_to_class_idx.keys()
            )
            self.attr_to_index = {name: idx for name, idx in zip(self.attrs_to_include, attrs_idx_list)}

            # Also construct Key -> Class Count
            attr_to_class_count = {name: train_ds.info.class_counts[idx] for name, idx in self.attr_to_index.items()}
        else:
            # Fallback: assume keys are indices 0..N
            self.attr_to_index = None
            attr_to_class_count = {i: c for i, c in enumerate(train_ds.info.class_counts)}

        # Get class occurrence counts (using generic key cache)
        self.attr_key_to_class_occurrence_counts = _get_attr_key_to_class_occurrence_counts(
            train_ds, self.cache_dir, attr_to_class_count, self.attr_to_index
        )

        # Configure Loss
        # Loss adapter expects indices
        self.loss_adapter = self._bind_loss_adapter(trainer.loss)
        if self.attr_to_index is not None:
            # Map keys (Names) to Indices
            attrs_idx = list(self.attr_to_index.values())
            self.loss_adapter.set_attrs_idx(attrs_idx)
        else:
            # Keys are already indices
            self.loss_adapter.set_attrs_idx(list(attr_to_class_count.keys()))

        # Initialize weights
        self.attr_key_to_class_weights = _calculate_attr_key_to_class_weights(self.attr_key_to_class_occurrence_counts)
        self._push_weights_to_loss()

        # Determine target validation split names
        if self.split_names is None:
            # Auto-detect by prefix
            self.target_split_names = [
                name for name in trainer.data.keys() if name.startswith(self.dataset_split_prefix)
            ]
        else:
            # Use explicit split names
            self.target_split_names = [name for name in self.split_names if name in trainer.data]
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

    def _push_weights_to_loss(self):
        """Translate key-based weights to index-based weights and push to loss."""
        if self.attr_key_to_class_weights is None:
            self.loss_adapter.set_class_weights(None)
            return

        if self.attr_to_index is not None:
            # Translate Key -> Index
            idx_weights = {self.attr_to_index[k]: w for k, w in self.attr_key_to_class_weights.items()}
            self.loss_adapter.set_class_weights(idx_weights)
        else:
            # Keys are already indices
            self.loss_adapter.set_class_weights(self.attr_key_to_class_weights)

    def _get_metric(self):
        """Lazily find the InternalMetricsProvider metric among the trainer's metrics."""
        if self._metric is None:
            self._metric = next((m for m in self._trainer.get_metrics()
                                 if isinstance(m, InternalMetricsProvider)), None)
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
        attr_key_to_recalls = {}
        for key, s in stats.items():
            tp = s["tp"].numpy()
            actual = s["actual"].numpy()
            # Handle zero division: if actual is 0, recall is 1 (matches zero_division=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tp / actual
                r[actual == 0] = 1.0
            attr_key_to_recalls[key] = r

        self.attr_key_to_class_weights = _calculate_attr_key_to_class_weights(
            self.attr_key_to_class_occurrence_counts, attr_key_to_recalls=attr_key_to_recalls
        )
        self._push_weights_to_loss()

    def _bind_loss_adapter(self, loss_callable):
        required_methods = ("set_attrs_idx", "set_class_weights")
        for method in required_methods:
            if not hasattr(loss_callable, method):
                raise TypeError(
                    "DynamicBalancedRecallWeights requires the configured loss to support "
                    f"{method}(). Use MultiAttributeCrossEntropyLoss or supply a compatible loss wrapper."
                )
            if not callable(getattr(loss_callable, method)):
                raise TypeError(f"DynamicBalancedRecallWeights expected trainer.loss.{method} to be callable.")
        return loss_callable

    def state_dict(self) -> dict:
        """Return state to be persisted in checkpoints."""
        return {
            "attr_key_to_class_weights": self.attr_key_to_class_weights,
            "attr_key_to_class_occurrence_counts": self.attr_key_to_class_occurrence_counts,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from a checkpoint."""
        # Clean backward compatibility: fallback to reading old keys if new keys missing
        self.attr_key_to_class_weights = state_dict.get(
            "attr_key_to_class_weights", state_dict.get("attr_idx_to_class_weights")
        )
        self.attr_key_to_class_occurrence_counts = state_dict.get(
            "attr_key_to_class_occurrence_counts", state_dict.get("attr_idx_to_class_occurrence_counts")
        )

        # Update the loss adapter with the restored weights
        if self.loss_adapter is not None:
            self._push_weights_to_loss()
