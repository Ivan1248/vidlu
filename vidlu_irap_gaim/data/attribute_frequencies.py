"""
Computation of attribute value frequencies from iRAP-BH dataset labels.

Used by baseline evaluation scripts and for empirical analysis of most common values.
"""

from collections import Counter
from dataclasses import dataclass
from vidlu.data import Dataset

import torch


@dataclass
class AttributeFrequencyStats:
    """Per-attribute frequency statistics computed from dataset labels."""

    most_common_class_indices: list[int]
    class_counts: tuple[int, ...]
    class_priors: list[torch.Tensor]
    class_occurrence_counts: list[torch.Tensor]


def compute_attribute_frequency_stats(dataset: Dataset) -> AttributeFrequencyStats:
    """Compute frequency statistics from a dataset with segment_id_to_labels.

    Uses all segments in segment_id_to_labels (not filtered by segment_ids).
    """
    all_labels = list(dataset.segment_id_to_labels.values())
    if not all_labels:
        raise ValueError("No labels found in dataset.")

    num_attrs = len(all_labels[0])
    class_counts = dataset.info.class_counts

    labels_per_attr = list(zip(*all_labels))

    most_common_class_indices = []
    class_priors = []
    class_occurrence_counts = []

    for attr_idx in range(num_attrs):
        k = class_counts[attr_idx]
        counts = Counter(labels_per_attr[attr_idx])

        most_common = counts.most_common(1)[0][0]
        most_common_class_indices.append(most_common)

        occurrence = torch.zeros(k, dtype=torch.long)
        for class_idx, count in counts.items():
            if class_idx < k:
                occurrence[class_idx] = count
        class_occurrence_counts.append(occurrence)

        prior = occurrence.float() / occurrence.sum()
        class_priors.append(prior)

    return AttributeFrequencyStats(
        most_common_class_indices=most_common_class_indices,
        class_counts=class_counts,
        class_priors=class_priors,
        class_occurrence_counts=class_occurrence_counts,
    )


def frequency_stats_to_attr_to_default_class_idx(
    stats: AttributeFrequencyStats,
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Convert frequency stats to an attr_to_default_class_idx dict for sparse schemes."""
    attr_names = list(attr_to_value_to_class_idx.keys())
    return dict(zip(attr_names, stats.most_common_class_indices))
