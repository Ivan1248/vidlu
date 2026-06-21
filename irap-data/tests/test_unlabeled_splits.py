"""Fixture-based tests for unlabeled-split support in `irap_data`.

Builds a small synthetic IRAP metadata directory on disk with both labeled and
unlabeled splits, then verifies that `make_bih_data` exposes all splits and
that unlabeled samples carry all-IGNORE targets.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import cv2

from irap_data import make_bih_data
from irap_data.attrs import filter_labeled_attrs
from irap_data.irap_dataset import IGNORE_LABEL_INDEX, IRAPDataset, MetaFiles


ATTRS = ["A", "B"]
IRAP_CODES = {"A": [10, 20], "B": [30, 40, 50]}  # codes per attribute


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _write_dummy_image(path: Path, *, w: int = 16, h: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), arr)


@pytest.fixture
def fake_dataset(tmp_path: Path):
    """Build a tiny IRAP metadata directory with labeled and unlabeled splits.

    Road sequence interleaves labeled and unlabeled segments to mirror the
    Vietnam prep pipeline's output (see build_metadata.py:402).
    """
    data_dir = tmp_path / "DATA"
    meta_dir = tmp_path / "DATA_METADATA"

    # 4 labeled (L0..L3) + 4 unlabeled (U0..U3) segments, interleaved.
    labeled = [f"L{i}" for i in range(4)]
    unlabeled = [f"U{i}" for i in range(4)]
    all_segs = ["L0", "U0", "L1", "U1", "L2", "U2", "L3", "U3"]

    # Write a 16x12 PNG per segment under DATA/images/.
    for sid in all_segs:
        _write_dummy_image(data_dir / "images" / f"{sid}.png")

    # attribute_metadata.json — defines attribute order and the full code space.
    _write_json(meta_dir / MetaFiles.ATTRIBUTE_METADATA, {
        "attribute_to_idx": {a: i for i, a in enumerate(ATTRS)},
        "attribute_value_to_irap_number": {
            a: {f"val_{c}": c for c in IRAP_CODES[a]} for a in ATTRS
        },
    })

    # segment_id_to_data_paths_rel.json — every segment (labeled + unlabeled).
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_DATA_PATHS, {
        sid: {"rgb": f"images/{sid}.png"} for sid in all_segs
    })

    # segment_id_to_road_data.json — only labeled segments carry attributes.
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA, {
        sid: {"required_attributes": {a: IRAP_CODES[a][0] for a in ATTRS}}
        for sid in labeled
    })

    # road_id_to_segment_id_sequence.json — single road, interleaved.
    _write_json(meta_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, {"road0": all_segs})

    # splits.json — labeled splits + unlabeled splits + unlocated.
    _write_json(meta_dir / MetaFiles.SPLITS, {
        "train": ["L0", "L1"],
        "val": ["L2"],
        "test": ["L3"],
        "unlabeled_train": ["U0", "U1"],
        "unlabeled_val": ["U2"],
        "unlabeled_test": ["U3"],
        "unlabeled_unlocated": [],
    })

    return data_dir, meta_dir


def _build_splits(fake_dataset):
    data_dir, meta_dir = fake_dataset
    # context_offsets=(0,) keeps the test independent of road-boundary edge cases.
    return make_bih_data(
        dataset_dir=data_dir,
        metadata_dir=meta_dir,
        context_offsets=(0,),
        use_ncontext_filter=False,
        input_dim_rgb=(16, 12, 3),
    )


def test_make_bih_data_exposes_unlabeled_splits(fake_dataset):
    splits = _build_splits(fake_dataset)
    # All keys from splits.json are present (the empty `unlabeled_unlocated`
    # split should still appear, just with length 0).
    assert set(splits.keys()) == {
        "train", "val", "test",
        "unlabeled_train", "unlabeled_val", "unlabeled_test", "unlabeled_unlocated",
    }


def test_unlabeled_split_targets_are_all_ignore(fake_dataset):
    splits = _build_splits(fake_dataset)
    u_train = splits["unlabeled_train"]
    assert len(u_train) == 2
    sample = u_train[0]
    assert sample["rgb"].ndim == 4  # (S, 3, H, W)
    assert sample["target"].shape == (len(ATTRS),)
    assert (sample["target"] == IGNORE_LABEL_INDEX).all()


def test_labeled_split_targets_are_class_indices(fake_dataset):
    splits = _build_splits(fake_dataset)
    train = splits["train"]
    sample = train[0]
    # First value in each attribute's code list -> class index 0.
    assert (sample["target"] == torch.tensor([0, 0])).all()


def test_unknown_subset_raises_with_available_keys(fake_dataset):
    data_dir, meta_dir = fake_dataset
    with pytest.raises(ValueError, match=r"Available in splits.json:.*unlabeled_train"):
        IRAPDataset(
            data_dir, subset="not_a_real_split",
            metadata_dir=meta_dir,
            context_offsets=(0,),
        )


def test_empty_unlabeled_split_is_loadable(fake_dataset):
    splits = _build_splits(fake_dataset)
    assert len(splits["unlabeled_unlocated"]) == 0


def test_attr_to_num_labeled_counts_labeled_samples(fake_dataset):
    splits = _build_splits(fake_dataset)
    # train = L0,L1, both attributes coded for both -> 2 labeled samples each.
    assert splits["train"].info.attr_to_num_labeled == {"A": 2, "B": 2}
    # Unlabeled splits carry all-ignore targets -> zero labeled samples.
    assert splits["unlabeled_train"].info.attr_to_num_labeled == {"A": 0, "B": 0}
    # With every attribute labeled, filtering is a no-op (the IRAP-BH case).
    assert filter_labeled_attrs(ATTRS, splits["train"].info.attr_to_num_labeled) == ("A", "B")


@pytest.fixture
def fake_dataset_uncoded_attr(tmp_path: Path):
    """Like ``fake_dataset`` but attribute ``B`` is never coded for any segment.

    Mirrors IRAP-Vietnam's BH-only attributes: ``B`` keeps a value vocabulary in
    ``attribute_metadata.json`` (so it survives the old vocabulary filter) yet has
    no labeled sample, so it would otherwise produce NaN metrics.
    """
    data_dir = tmp_path / "DATA"
    meta_dir = tmp_path / "DATA_METADATA"
    labeled = [f"L{i}" for i in range(4)]
    for sid in labeled:
        _write_dummy_image(data_dir / "images" / f"{sid}.png")
    _write_json(meta_dir / MetaFiles.ATTRIBUTE_METADATA, {
        "attribute_to_idx": {a: i for i, a in enumerate(ATTRS)},
        "attribute_value_to_irap_number": {
            a: {f"val_{c}": c for c in IRAP_CODES[a]} for a in ATTRS
        },
    })
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_DATA_PATHS, {
        sid: {"rgb": f"images/{sid}.png"} for sid in labeled
    })
    # Only attribute "A" is coded; "B" is absent -> mapped to the ignore index.
    _write_json(meta_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA, {
        sid: {"required_attributes": {"A": IRAP_CODES["A"][0]}} for sid in labeled
    })
    _write_json(meta_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, {"road0": labeled})
    _write_json(meta_dir / MetaFiles.SPLITS, {
        "train": ["L0", "L1"], "val": ["L2"], "test": ["L3"],
    })
    return data_dir, meta_dir


def test_filter_labeled_attrs_drops_uncoded_attribute(fake_dataset_uncoded_attr):
    data_dir, meta_dir = fake_dataset_uncoded_attr
    train = make_bih_data(
        dataset_dir=data_dir, metadata_dir=meta_dir,
        context_offsets=(0,), use_ncontext_filter=False,
        input_dim_rgb=(16, 12, 3), allow_missing_attributes=True,
    )["train"]
    # "B" has a value vocabulary but no labeled sample.
    assert train.info.attr_to_value_to_class_idx.get("B")  # vocabulary present
    assert train.info.attr_to_num_labeled == {"A": 2, "B": 0}
    # The label-based filter drops "B" (the old vocabulary filter would keep it).
    assert filter_labeled_attrs(ATTRS, train.info.attr_to_num_labeled) == ("A",)
