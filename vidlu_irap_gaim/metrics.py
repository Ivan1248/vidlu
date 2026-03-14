from typing import Sequence, Protocol, runtime_checkable

import torch

from vidlu.metrics import AccumulatingMetric, ClassificationMetrics
from vidlu.utils.collections import NameDict

import numpy as np


@runtime_checkable
class InternalMetricsProvider(Protocol):
    """Protocol for metrics that provide internal metrics/statistics for extensions.

    Metrics implementing this protocol can be discovered by extensions that need
    internal data (e.g., class-level recall/precision statistics for dynamic weight computation).
    """

    def get_internal_metrics(self) -> dict[int, dict[str, torch.Tensor]]: ...


class MultiAttributeClassificationMetrics(AccumulatingMetric):
    """
    Multi-attribute classification metrics.

    Metrics naming convention:
        - a{X}: average of {X} across attributes (returns scalar for scalar base metrics)
        - {X}: per-attribute {X} (returns dict mapping attribute index to value)

    Where {X} is any base metric from ClassificationMetrics:
        - A: Accuracy (scalar)
        - mP, mR, mF1, mIoU: Macro-averaged Precision/Recall/F1/IoU (scalars)
        - P, R, F1, IoU: Per-class metrics (arrays)

    Return format:
        The returned dictionary uses the exact metric names from `self.metrics`.
        - For average metrics (a{X}): returns a scalar value (averaged across attributes)
        - For per-attribute metrics ({X}): returns a dict {attr_index: value}
        - Attempting to average per-class metrics (aP, aR, aF1, aIoU) raises ValueError

    Examples:
        With metrics=('amF1', 'amP', 'amR', 'mF1', 'A'):
        {
            'amF1': 0.85,                   # scalar - average F1
            'amP': 0.85,                    # scalar - average precision
            'amR': 0.85,                    # scalar - average recall
            'mF1': {0: 0.70, 1: 0.74, ...}, # dict - per-attribute macro F1
            'A': {0: 0.84, 1: 0.86, ...},   # dict - per-attribute accuracy
        }
    """

    # Base metrics supported by ClassificationMetrics
    BASE_METRICS = {"A", "mP", "mR", "mF1", "mIoU", "P", "R", "F1", "IoU"}

    def __init__(
        self,
        attr_to_class_count: dict[object, int] | list[int] | None = None,
        attrs_to_include: Sequence[object] = None,
        attr_to_index: dict[object, int] | None = None,
        metrics: Sequence[str] = ("amF1", "amP", "amR"),
        only_present_classes: bool = False,
    ):
        """
        Args:
            class_counts: Tuple of class counts per attribute.
            attrs: Sequence of attribute indices to compute metrics for.
            metrics: Sequence of metric names to compute. Can include:
                - Base metrics: A, mP, mR, mF1, mIoU, P, R, F1, IoU
                - Average metrics: aA, amP, amR, amF1, amIoU (for scalar base metrics)
        """
        self.attr_to_class_count = (
            attr_to_class_count if isinstance(attr_to_class_count, dict) else dict(enumerate(attr_to_class_count))
        )
        self.attrs = list(attrs_to_include)
        self.attr_to_index = attr_to_index  # If None, assumes keys are indices
        self.metrics = metrics

        # Determine base metrics needed for ClassificationMetrics
        def _get_base_metric(m):
            clean = m.lstrip("_")
            if clean in self.BASE_METRICS:
                return clean
            if clean.startswith("a") and clean[1:] in self.BASE_METRICS:
                return clean[1:]
            return None

        base_metrics = set()
        for m in metrics:
            base = _get_base_metric(m)
            if base:
                base_metrics.add(base)

        self.attr_to_metrics = {}
        if self.attr_to_class_count is not None:
            for a in self.attrs:
                if a not in self.attr_to_class_count:
                    continue
                # We use a custom get_target/get_hard_prediction because we'll pass sliced data
                self.attr_to_metrics[a] = ClassificationMetrics(
                    class_count=self.attr_to_class_count[a],
                    get_target=lambda r: r["target"],
                    get_hard_prediction=lambda r: r["out"].argmax(1),
                    metrics=tuple(base_metrics),
                    only_present_classes=only_present_classes,
                )
        self.reset()

    def reset(self):
        for m in self.attr_to_metrics.values():
            m.reset()

    @torch.no_grad()
    def update(self, iter_result):
        outs = iter_result.out  # tuple of (B, K_i)
        true = iter_result.target  # (B, A)
        if outs is None:
            return
        for key, m in self.attr_to_metrics.items():
            # Determine tensor index
            if self.attr_to_index is not None:
                if key not in self.attr_to_index:
                    continue
                idx = self.attr_to_index[key]
            else:
                # Fallback: assume key is the index
                idx = key if isinstance(key, int) else -1

            if idx >= 0 and idx < len(outs):
                m.update(NameDict(target=true[:, idx], out=outs[idx]))

    def get_internal_metrics(self) -> dict[int, dict[str, torch.Tensor]]:
        """Returns internal metrics/statistics for programmatic consumption.

        This method is always available and does not require any special metric names
        in the metrics list. Statistics are computed from the current confusion matrices.

        Returns:
            Dictionary mapping attribute index to stats dict with keys:
            - 'tp': true positives per class (tensor, on CPU)
            - 'pos': predicted positives per class (tensor, on CPU)
            - 'actual': actual positives per class (tensor, on CPU)
        """
        stats = {}
        for a, m in self.attr_to_metrics.items():
            # m.cm is on device, move to cpu
            cm = m.cm.cpu()
            tp = cm.diagonal()
            actual = cm.sum(1)
            pos = cm.sum(0)
            stats[a] = {"tp": tp, "pos": pos, "actual": actual}
        return stats

    @torch.no_grad()
    def compute(self, metrics=None):
        """
        Computes metrics for all attributes.

        Args:
            metrics: Optional list of metrics to compute, overrides `self.metrics`. If None,
            computes all metrics.

        Returns:
            Dictionary with keys matching `self.metrics`. Values are:
            - Scalars for average metrics (a{X})
            - Dicts mapping attribute indices to values for per-attribute metrics ({X})
        """
        attr_to_metrics = {a: m.compute() for a, m in self.attr_to_metrics.items()}

        results = {}
        for metric_key in self.metrics if metrics is None else metrics:
            clean = metric_key.lstrip("_")
            metric_type = None  # 'per_attr' or 'average'
            base_metric = None

            if clean in self.BASE_METRICS:
                metric_type = "per_attr"
                base_metric = clean
            elif clean.startswith("a") and clean[1:] in self.BASE_METRICS:
                metric_type = "average"
                base_metric = clean[1:]

            if metric_type == "per_attr":
                # Per-attribute metric: output dict mapping attr index to value
                results[metric_key] = {
                    a: attr_results[base_metric]
                    for a, attr_results in attr_to_metrics.items()
                    if base_metric in attr_results
                }
            elif metric_type == "average":
                # Average metric across attributes (e.g., 'amF1' -> average of 'mF1')
                values = [m.get(base_metric) for m in attr_to_metrics.values() if base_metric in m]
                if values:
                    # Handle both scalar and array values
                    if np.isscalar(values[0]) or (isinstance(values[0], np.ndarray) and values[0].ndim == 0):
                        results[metric_key] = np.mean(values)
                    else:
                        raise ValueError(f"Cannot average per-class metrics across attributes: {base_metric}")
                else:
                    results[metric_key] = 0.0

        return results


class MultiAttributeAccuracy(AccumulatingMetric):
    def __init__(
        self,
        name: str = "acc",
        attrs_to_include: Sequence[object] | None = None,
        attr_to_index: dict[object, int] | None = None,
    ):
        self.name = name
        self.attrs = list(attrs_to_include) if attrs_to_include is not None else None
        self.attr_to_index = attr_to_index
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, iter_result):
        outs = iter_result.out
        true = iter_result.target
        if outs is None:
            return
        if self.attrs is None:
            raise RuntimeError("MultiAttributeAccuracy.attrs_to_include must be set.")

        keys_to_update = self.attrs
        matched = 0
        for key in keys_to_update:
            # Determine tensor index
            if self.attr_to_index is not None:
                if key not in self.attr_to_index:
                    continue
                idx = self.attr_to_index[key]
            else:
                # Fallback: assume key is the index
                idx = key if isinstance(key, int) else -1

            if idx < 0 or idx >= len(outs):
                continue
            pred = outs[idx].argmax(1)
            matched += torch.sum(pred == true[:, idx])
        self.correct += matched.item() if isinstance(matched, torch.Tensor) else matched
        self.total += true.shape[0] * len(keys_to_update)

    @torch.no_grad()
    def compute(self):
        return {self.name: (self.correct / max(1, self.total))}


def get_irap_metrics(
    dataset=None,
    class_counts: tuple[int, ...] | None = None,
    attrs_to_include: tuple[str, ...] | None = None,
):
    """
    Creates IRAP metrics configured with attrs_to_include filtering.

    Args:
        dataset: Dataset with info.attribute_names (used to map attribute names to indices).
        class_counts: Optional tuple of class counts per attribute. If None, uses dataset.info.class_counts.
        attrs_to_include: Optional tuple of attribute names. If None, uses canonical paper subset.

    Returns:
        Tuple of (MultiAttributeClassificationMetrics, MultiAttributeAccuracy) configured with attrs_idx.
    """
    from .attrs import get_attrs_to_include, map_attr_names_to_indices

    if dataset is None:
        from .datasets import make_bih_data

        dataset = make_bih_data()["train"]

    if class_counts is None:
        if not hasattr(dataset, "info") or not hasattr(dataset.info, "class_counts"):
            raise ValueError("Dataset must have info.class_counts or class_counts must be provided")
        class_counts = dataset.info.class_counts

    if attrs_to_include is None:
        attrs_to_include = get_attrs_to_include()

    attrs_idx_list = map_attr_names_to_indices(attrs_to_include, dataset.info.attr_to_value_to_class_idx.keys())

    # Construct mappings for name-based logic
    # We rely on map_attr_names_to_indices behavior or dataset.info structure to get meaningful names
    # Actually, attrs_to_include ARE the names we want to use.
    # We construct proper mapping: Name -> Index
    attr_to_index = {name: idx for name, idx in zip(attrs_to_include, attrs_idx_list)}

    # Construct class counts keyed by name
    # We need to map Name -> Class Count. class_counts tuple is ordered by GLOBAL indices?
    # No, class_counts tuple usually corresponds to dataset.attribute_names order.
    # attrs_idx_list contains indices into that tuple.
    attr_to_class_count = {name: class_counts[idx] for name, idx in attr_to_index.items() if idx < len(class_counts)}

    ma_metrics = ("amF1", "amP", "amR")
    # Enable per-attribute metrics by default for better visibility
    # Use "_" prefix to hide from standard logger but keep available for printer
    ma_metrics = (*ma_metrics, "_mF1", "_A")

    return (
        # only_present=True is for excluding non-present classes like in Kačan et al. (2025)
        MultiAttributeClassificationMetrics(
            metrics=ma_metrics,
            attr_to_class_count=attr_to_class_count,
            attrs_to_include=attrs_to_include,
            attr_to_index=attr_to_index,
            only_present_classes=True,
        ),
        MultiAttributeAccuracy(attrs_to_include=attrs_to_include, attr_to_index=attr_to_index),
    )
