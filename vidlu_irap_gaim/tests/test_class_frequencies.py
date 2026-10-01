"""Tests of the class frequencies behind the sparse VLM defaults and the baselines."""

from types import SimpleNamespace

from vidlu_irap_gaim.class_frequencies import (
    compute_attr_to_class_counts, compute_attr_to_most_common_class_idx)

IGNORE = -1


def _info():
    """The ignore label is the most frequent value of attribute A, and B is never labeled.
    s4 and s5 have labels but are not examples, as for segments without a complete context
    window."""
    labels = {"s0": [2, IGNORE], "s1": [IGNORE, IGNORE], "s2": [IGNORE, IGNORE],
              "s3": [IGNORE, IGNORE], "s4": [0, IGNORE], "s5": [0, IGNORE]}
    return SimpleNamespace(segment_ids=["s0", "s1", "s2", "s3"], segment_id_to_labels=labels,
                           class_counts=(3, 2),
                           attr_to_value_to_class_idx={"A": {"a": 0, "b": 1, "c": 2},
                                                       "B": {"x": 0, "y": 1}})


def test_counts_cover_all_labeled_segments_and_skip_the_ignore_label():
    counts = compute_attr_to_class_counts(_info())
    assert {attr: c.tolist() for attr, c in counts.items()} == {"A": [2, 0, 1], "B": [0, 0]}


def test_most_common_class_is_a_class():
    assert compute_attr_to_most_common_class_idx(_info()) == {"A": 0, "B": 0}
