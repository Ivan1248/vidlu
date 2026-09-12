"""Dynamic balanced-recall class weights for multi-attribute classification.

A port of ``calculate_new_class_weights`` from ``libs/irap_gaim-orig/train_local_rec.py:127-169``:
weights attribute cross-entropy as

    w_c = inverse_frequency_c * (1 - recall_c) + sqrt(inverse_frequency_c) * recall_c,

where inverse frequencies derive from training class counts and recalls derive from
the previous epoch's validation confusion matrix. High-recall classes are damped
toward ``sqrt(inverse_frequency)``, while low-recall classes retain the full inverse frequency.
"""

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from irap_data.attrs import get_attrs_to_include, map_attr_names_to_indices
from irap_data.irap_dataset import IGNORE_LABEL_INDEX, compute_label_matrix

from vidlu.metrics import confusion_matrix_class_stats
from vidlu.training.extensions import TrainerExtension

log = logging.getLogger(__name__)


@runtime_checkable
class ConfusionMatricesProvider(Protocol):
    """A metric exposing one confusion matrix per attribute, rows = ground truth."""

    def get_confusion_matrices(self) -> dict[object, torch.Tensor]: ...

# Inverse frequency substituted for a class with no training examples, from the original
# (``train_local_rec.py:152``). Such a class never occurs as a target, so cross-entropy never
# reads its weight – provided the counts cover the same data the loss runs on, which is
# what ``_get_train_datasets`` enforces.
INVERSE_FREQUENCY_FOR_ABSENT_CLASS = 1e-4


def compute_attr_to_class_occurrence_counts(
    datasets: Sequence, attr_to_index: dict[str, int], attr_to_num_classes: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Counts class occurrences per attribute, pooled over ``datasets``.

    Corresponds to ``attribute_class_idx_to_occurrences`` in ``libs/irap_gaim-orig``.
    Excludes the ignore label (``IGNORE_LABEL_INDEX``): because negative indices are
    valid in Python arrays, counting it would silently assign unannotated samples
    to the final class and distort the inverse frequency baseline.

    Args:
        datasets: iRAP datasets whose `info` carries `segment_ids` and
            `segment_id_to_labels`. Counts are summed over all of them, so joint training
            on several `train*` splits gets the priors of their union.
        attr_to_index: Maps an attribute name to its column in the label matrix.
        attr_to_num_classes: Maps an attribute name to its number of classes.

    Returns:
        Maps an attribute name to an int64 tensor of length `attr_to_num_classes[attr]`,
        whose element `c` is the number of examples labelled with class `c`.
    """
    counts = {attr: np.zeros(n, dtype=np.int64) for attr, n in attr_to_num_classes.items()}
    for dataset in datasets:
        info = dataset.info
        labels = compute_label_matrix(info.segment_id_to_labels, info.segment_ids,
                                      len(info.class_counts))
        for attr, num_classes in attr_to_num_classes.items():
            column = labels[:, attr_to_index[attr]]
            observed = column[column != IGNORE_LABEL_INDEX]
            if observed.size and (observed.min() < 0 or observed.max() >= num_classes):
                invalid = observed[(observed < 0) | (observed >= num_classes)]
                raise ValueError(
                    f"Attribute '{attr}' has class indices outside [0, {num_classes}) and"
                    f" other than the ignore label {IGNORE_LABEL_INDEX}:"
                    f" {sorted(set(invalid.tolist()))}.")
            counts[attr] += np.bincount(observed, minlength=num_classes)
    return {attr: torch.from_numpy(c) for attr, c in counts.items()}


def _calculate_attr_to_class_weights(
    attr_to_class_occurrence_counts: dict[str, torch.Tensor],
    attr_to_class_recalls: dict[str, np.ndarray] | None = None,
) -> dict[str, torch.Tensor]:
    """Calculates class weights for the dynamic balanced-recall loss.

    This corresponds to `calculate_new_class_weights` in the original iRAP GAIM code.

    Args:
        attr_to_class_occurrence_counts: Maps an attribute name to its per-class training
            occurrence counts.
        attr_to_class_recalls: Maps an attribute name to its per-class validation recalls.
            `None` before any evaluation has run, in which case a random classifier's
            recalls (`1 / num_classes`) are assumed, as in the original. An attribute the
            validation metric does not cover gets recall 1, the same convention as a class
            with no validation examples.

    Returns:
        Maps an attribute name to a float32 weight tensor, for every attribute in
        `attr_to_class_occurrence_counts`.
    """
    attr_to_class_weights = {}
    for attr, occurrence_counts in attr_to_class_occurrence_counts.items():
        num_classes = len(occurrence_counts)
        if attr_to_class_recalls is None:
            recalls = np.full(num_classes, 1 / num_classes, dtype=np.float64)
        else:
            recalls = np.asarray(attr_to_class_recalls.get(attr, np.ones(num_classes)),
                                 dtype=np.float64)

        occurrences = occurrence_counts.numpy().astype(np.float64)
        total = occurrences.sum()
        inverse_frequencies = np.where(occurrences > 0, total / np.maximum(occurrences, 1),
                                       INVERSE_FREQUENCY_FOR_ABSENT_CLASS)
        weights = (inverse_frequencies * (1.0 - recalls)
                   + np.sqrt(inverse_frequencies) * recalls)
        attr_to_class_weights[attr] = torch.tensor(weights, dtype=torch.float32)
    return attr_to_class_weights


def _compute_attr_to_class_recalls(
    attr_to_class_stats: dict[str, dict[str, torch.Tensor]]
) -> dict[str, np.ndarray]:
    """Per-class recalls from confusion-matrix statistics.

    A class with no validation examples gets recall 1, matching the original's
    `recall_score(..., zero_division=1)`.
    """
    attr_to_class_recalls = {}
    for attr, stats in attr_to_class_stats.items():
        true_positives = stats["tp"].numpy().astype(np.float64)
        actual = stats["actual"].numpy().astype(np.float64)
        attr_to_class_recalls[attr] = np.where(actual == 0, 1.0,
                                               true_positives / np.maximum(actual, 1))
    return attr_to_class_recalls


class DynamicBalancedRecallWeights(TrainerExtension):
    """Re-weights each attribute's cross-entropy from training priors and validation recalls.

    Computes training class occurrence counts once over all ``train*`` splits, so that joint
    training uses the prior distribution of their union. After each evaluation of the tracked
    validation split(s), reads per-class recalls from the metric belonging to that split, then
    recomputes and applies the loss weights.

    Args:
        attrs_to_include: Attribute names to weight; None (the default) means the canonical
            attributes (`irap_data.attrs.get_attrs_to_include`). Attributes with no labelled
            example in the training data are dropped from this set, since neither their
            priors nor their recalls are defined (e.g. iRAP-Vietnam's seven BH-only
            attributes).
        val_split_prefix: Prefix identifying the validation splits. Ignored when
            `recall_split_names` is given.
        train_split_names: Training splits to take occurrence counts from. `None` (the
            default) uses every `trainer.data` key starting with "train", which is what
            `Trainer.get_training_data_loader` trains on.
        recall_split_names: Validation splits to take recalls from. `None` (the default)
            uses the *first* split matching `val_split_prefix`. Naming several splits
            pools their confusion-matrix statistics within each epoch.

    Note:
        Requires a `ConfusionMatricesProvider` metric (e.g.
        `MultiAttributeClassificationMetrics`) among `trainer.metrics`, and a loss
        supporting `set_attrs_idx()` / `set_class_weights()` (i.e.
        `MultiAttributeCrossEntropyLoss`).
    """

    def __init__(
        self,
        attrs_to_include: Sequence[str] | None = None,
        val_split_prefix: str = "val",
        train_split_names: Sequence[str] | None = None,
        recall_split_names: Sequence[str] | None = None,
    ):
        self.attrs_to_include = tuple(
            get_attrs_to_include() if attrs_to_include is None else attrs_to_include)
        self.val_split_prefix = val_split_prefix
        self.train_split_names = None if train_split_names is None else list(train_split_names)
        self.recall_split_names = None if recall_split_names is None else list(recall_split_names)
        self.attr_to_class_occurrence_counts: dict[str, torch.Tensor] | None = None
        self.attr_to_class_weights: dict[str, torch.Tensor] | None = None
        self.attr_to_index: dict[str, int] | None = None
        self.target_split_names: list[str] | None = None
        self.loss = None
        self._trainer = None
        self._split_name_to_metric: dict[str, ConfusionMatricesProvider] = {}
        self._pooled_attr_to_class_stats: dict[str, dict[str, torch.Tensor]] = {}
        self._reported_attrs_without_recalls = False

    def initialize(self, trainer):
        if trainer.data is None:
            raise ValueError("DynamicBalancedRecallWeights requires trainer.data to be set.")
        self._trainer = trainer

        train_datasets = self._get_train_datasets(trainer)
        reference_info = next(iter(train_datasets.values())).info
        self.attr_to_index = dict(zip(
            self.attrs_to_include,
            map_attr_names_to_indices(self.attrs_to_include,
                                      list(reference_info.attr_to_value_to_class_idx.keys()))))
        attr_to_num_classes = {attr: reference_info.class_counts[i]
                               for attr, i in self.attr_to_index.items()}

        counts = compute_attr_to_class_occurrence_counts(
            list(train_datasets.values()), self.attr_to_index, attr_to_num_classes)
        # An attribute a release does not annotate has every label ignored, so it has no
        # priors and no recalls. Dropping it here keeps this set equal to the metric's,
        # which `get_irap_metrics` derives the same way via `filter_labeled_attrs`.
        unlabeled = [attr for attr, c in counts.items() if c.sum() == 0]
        if unlabeled:
            log.info(f"DynamicBalancedRecallWeights: not weighting {len(unlabeled)} attributes"
                     f" with no labelled training example: {', '.join(unlabeled)}.")
            for attr in unlabeled:
                del counts[attr], self.attr_to_index[attr]
        if not counts:
            raise ValueError(
                "DynamicBalancedRecallWeights: none of the attributes"
                f" {list(self.attrs_to_include)} has a labelled training example.")
        self.attr_to_class_occurrence_counts = counts

        self.loss = self._check_loss_supports_class_weights(trainer.loss)
        self.loss.set_attrs_idx(list(self.attr_to_index.values()))
        self.attr_to_class_weights = _calculate_attr_to_class_weights(counts)
        self._set_loss_class_weights()

        self.target_split_names = self._get_target_split_names(trainer)
        log.info(f"DynamicBalancedRecallWeights: class priors from"
                 f" {', '.join(f'{n} ({len(ds)})' for n, ds in train_datasets.items())};"
                 f" recalls from {', '.join(self.target_split_names)}.")

        @trainer.evaluation.epoch_completed.handler
        def on_evaluation_epoch_completed(state):
            split_name = getattr(state, "split_name", None)
            if split_name in self.target_split_names:
                self._update_class_weights(split_name)

    def _get_train_datasets(self, trainer) -> dict:
        """The training splits the occurrence counts are taken over, validated."""
        if self.train_split_names is None:
            names = [name for name in trainer.data if name.startswith("train")]
            if not names:
                raise ValueError(
                    "DynamicBalancedRecallWeights found no training split in trainer.data"
                    f" (expected a key starting with 'train'); got {list(trainer.data)}.")
        else:
            names = self.train_split_names
            if missing := [n for n in names if n not in trainer.data]:
                raise ValueError(
                    f"DynamicBalancedRecallWeights: train splits {missing} are not in"
                    f" trainer.data. Available: {list(trainer.data)}.")

        datasets = {name: trainer.data[name] for name in names}
        for name, dataset in datasets.items():
            num_described = len(dataset.info.segment_ids)
            if num_described != len(dataset):
                raise ValueError(
                    f"Split '{name}' has {len(dataset)} examples but its info describes"
                    f" {num_described}, so class priors computed from it would not match the"
                    f" data the loss sees. This happens when datasets are joined"
                    f" (`a.join(b, info=b.info)` keeps only one info) or subsetted. Pass the"
                    f" splits separately instead of joining them.")
        return datasets

    def _get_target_split_names(self, trainer) -> list[str]:
        """The validation splits whose recalls drive the weight updates."""
        if self.recall_split_names is None:
            matching = [name for name in trainer.data
                        if name.startswith(self.val_split_prefix)]
            if not matching:
                raise ValueError(
                    "DynamicBalancedRecallWeights found no validation split in trainer.data"
                    f" starting with '{self.val_split_prefix}'; got {list(trainer.data)}.")
            # Only the first: evaluation runs per split, and each split's metric is reset
            # once reported, so reacting to every one would let the last split's (or an
            # already-reset metric's) recalls silently overwrite the rest.
            return matching[:1]

        if missing := [n for n in self.recall_split_names if n not in trainer.data]:
            raise ValueError(
                f"DynamicBalancedRecallWeights: recall splits {missing} are not in"
                f" trainer.data. Available: {list(trainer.data)}.")
        return list(self.recall_split_names)

    def _get_confusion_matrices_provider(self, split_name) -> ConfusionMatricesProvider:
        """The metric providing the per-attribute confusion matrices for `split_name`.

        Resolved per split: `trainer.metrics` may be a `{split_name: metrics}` mapping, and
        asking without the split name would always return the first entry's metrics, whose
        confusion matrix belongs to a different split (and has since been reset).
        """
        if split_name not in self._split_name_to_metric:
            metric = next((m for m in self._trainer.get_metrics(split_name)
                           if isinstance(m, ConfusionMatricesProvider)), None)
            if metric is None:
                raise RuntimeError(
                    f"DynamicBalancedRecallWeights: no metric with get_confusion_matrices() for"
                    f" split '{split_name}'. Ensure MultiAttributeClassificationMetrics is"
                    f" included in the metrics for it.")
            self._split_name_to_metric[split_name] = metric
        return self._split_name_to_metric[split_name]

    def _update_class_weights(self, split_name):
        """Recomputes the weights from the just-completed evaluation of `split_name`."""
        attr_to_class_stats = {
            attr: confusion_matrix_class_stats(cm.cpu())
            for attr, cm in self._get_confusion_matrices_provider(split_name)
            .get_confusion_matrices().items()}
        if not attr_to_class_stats:
            raise RuntimeError(
                f"DynamicBalancedRecallWeights: the metric for split '{split_name}' returned no"
                f" statistics. Ensure it has been updated with evaluation data.")

        if not self._reported_attrs_without_recalls:
            self._reported_attrs_without_recalls = True
            if missing := [a for a in self.attr_to_class_occurrence_counts
                           if a not in attr_to_class_stats]:
                log.warning(f"DynamicBalancedRecallWeights: the metric for '{split_name}' does"
                            f" not cover {', '.join(missing)}, which do have training class"
                            f" priors. They are weighted as if perfectly recalled.")

        if split_name == self.target_split_names[0]:
            self._pooled_attr_to_class_stats = {}  # a new epoch's pass over the splits
        for attr, stats in attr_to_class_stats.items():
            pooled = self._pooled_attr_to_class_stats.get(attr)
            self._pooled_attr_to_class_stats[attr] = (
                dict(stats) if pooled is None
                else {k: pooled[k] + v for k, v in stats.items()})

        self.attr_to_class_weights = _calculate_attr_to_class_weights(
            self.attr_to_class_occurrence_counts,
            _compute_attr_to_class_recalls(self._pooled_attr_to_class_stats))
        self._set_loss_class_weights()

    def _set_loss_class_weights(self):
        """Pushes the current weights into the loss, keyed by global attribute index."""
        self.loss.set_class_weights(
            {self.attr_to_index[attr]: w for attr, w in self.attr_to_class_weights.items()})

    @staticmethod
    def _check_loss_supports_class_weights(loss):
        for method in ("set_attrs_idx", "set_class_weights"):
            if not callable(getattr(loss, method, None)):
                raise TypeError(
                    "DynamicBalancedRecallWeights requires the configured loss to support"
                    f" {method}(). Use MultiAttributeCrossEntropyLoss or a compatible wrapper.")
        return loss

    def state_dict(self) -> dict:
        # Only the weights: they carry the previous epoch's recalls and cannot be
        # recovered from the data, whereas the occurrence counts are recomputed in
        # `initialize` -- persisting those would let a checkpoint restore stale counts
        # over freshly computed ones.
        return {"attr_to_class_weights": self.attr_to_class_weights}

    def load_state_dict(self, state_dict: dict):
        if "attr_to_class_weights" not in state_dict:
            raise KeyError(
                f"The DynamicBalancedRecallWeights state holds {sorted(state_dict)} but no"
                f" 'attr_to_class_weights'. It was not written by this version of the extension;"
                f" start the run from scratch instead of resuming.")
        self.attr_to_class_weights = state_dict["attr_to_class_weights"]
        if self.loss is not None:
            self._set_loss_class_weights()
