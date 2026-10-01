"""Class frequencies of a split, for the defaults of sparse VLM responses and for baselines."""

import numpy as np

from irap_data import compute_class_occurrence_counts


def compute_attr_to_class_counts(info) -> dict[str, np.ndarray]:
    """Counts the segments of each class of each attribute, over all labeled segments of a split.

    The counts cover every segment in `info.segment_id_to_labels`, not only the examples in
    `info.segment_ids`, so they do not depend on the context offsets or the N-context filter.

    Returns:
        Maps each attribute name, in schema order, to its (K,) class counts.
    """
    return compute_class_occurrence_counts(info, segment_ids=list(info.segment_id_to_labels))


def compute_attr_to_most_common_class_idx(info) -> dict[str, int]:
    """The most common class of each attribute (see `compute_attr_to_class_counts`).

    Unlabeled segments are not counted. An attribute without labels gets class 0.
    """
    return {attr: int(np.argmax(counts))
            for attr, counts in compute_attr_to_class_counts(info).items()}
