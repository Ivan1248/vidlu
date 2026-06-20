import torch
import numpy as np
import tempfile, os

from vidlu_irap_gaim.training.semisup import (
    multi_attribute_kl_div_ll,
    make_semisup_data,
    make_pseudo_labeled_data,
    PseudoLabeledDataset,
)
from irap_data import Dataset


class MockDataset(Dataset):
    def __init__(self, length=100):
        super().__init__()
        self._length = length
        self.data = list(range(length))
        self.info = {"name": "mock"}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.data[idx]


def test_multi_attribute_kl_div_ll():
    # Create fake logits for 2 attributes, batch size 3
    # Perfect match should yield 0 loss
    p1 = torch.randn(3, 5)
    p2 = torch.randn(3, 4)

    # Target logits same as predictions
    loss = multi_attribute_kl_div_ll((p1, p2), (p1, p2))
    assert torch.isclose(loss, torch.tensor(0.0))

    # Mismatch should yield positive loss
    t1 = torch.randn(3, 5)
    t2 = torch.randn(3, 4)
    loss_mismatch = multi_attribute_kl_div_ll((p1, p2), (t1, t2))
    assert loss_mismatch > 0

    # Test reduction='none'
    loss_none = multi_attribute_kl_div_ll((p1, p2), (t1, t2), reduction="none")
    assert loss_none.shape == (3,)


def test_make_semisup_data_splitting():
    base_data = {"train": MockDataset(100), "val": MockDataset(10), "test": MockDataset(10)}

    # 10% labeled -> 10 labeled, 90 unlabeled
    data = make_semisup_data(base_data, labeled_ratio=0.1, seed=42)

    # data["train"] is the labeled dataset, data["train_u"] is the unlabeled dataset
    train_l = data["train"]
    train_u = data["train_u"]
    assert len(train_l) == 10
    assert len(train_u) == 90

    # Verify disjointness (mock data is indices 0-99).
    # Since permute/split wraps in SubDataset ops, we just iterate to get indices.
    l_indices = set(list(train_l))
    u_indices = set(list(train_u))

    assert len(l_indices.intersection(u_indices)) == 0
    assert len(l_indices) + len(u_indices) == 100


def test_make_semisup_data_uses_real_unlabeled_split():
    # When splits.json carries an `unlabeled_train` key (e.g. IRAP-Vietnam),
    # make_semisup_data should use it as `train_u` and leave the labeled
    # `train` split whole, ignoring labeled_ratio.
    base_data = {
        "train": MockDataset(100),
        "val": MockDataset(10),
        "test": MockDataset(10),
        "unlabeled_train": MockDataset(73),
    }

    data = make_semisup_data(base_data, labeled_ratio=0.1, seed=42)
    assert data["train"] is base_data["train"]
    assert data["train_u"] is base_data["unlabeled_train"]
    assert len(data["train"]) == 100
    assert len(data["train_u"]) == 73

    # Opt-out flag forces the synthetic-split path.
    data2 = make_semisup_data(base_data, labeled_ratio=0.1, seed=42, prefer_real_unlabeled=False)
    assert len(data2["train"]) == 10
    assert len(data2["train_u"]) == 90


def test_make_semisup_data_no_shuffle():
    # Test that shuffle=False takes the first portion without shuffling
    base_data = {"train": MockDataset(100), "val": MockDataset(10), "test": MockDataset(10)}

    # 10% labeled without shuffling -> first 10 samples
    data = make_semisup_data(base_data, labeled_ratio=0.1, shuffle=False)

    train_l = data["train"]
    train_u = data["train_u"]
    assert len(train_l) == 10
    assert len(train_u) == 90

    # Without shuffling, labeled should be first 10 indices (0-9)
    l_indices = list(train_l)
    assert l_indices == list(range(10))


class MockRecordDataset(Dataset):
    """MockDataset that returns dict items (compatible with Record wrapping)."""

    def __init__(self, length=100):
        super().__init__()
        self._length = length
        self.info = {"name": "mock"}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {"x": idx}


def test_make_pseudo_labeled_data():
    base_data = {"train": MockRecordDataset(100), "val": MockRecordDataset(10), "test": MockRecordDataset(10)}
    # Write a tiny .npz with 90 pseudo-labels (unlabeled portion = 90)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        npz_path = f.name
    try:
        np.savez_compressed(npz_path, labels=np.zeros((90, 2), dtype=np.int16))
        data = make_pseudo_labeled_data(
            base_data, labeled_ratio=0.1, pseudo_labels_path=npz_path, seed=42
        )
        assert "train" in data
        assert "train_u" in data
        assert "val" in data
        assert "test" in data
        assert len(data["train"]) == 10
        assert len(data["train_u"]) == 90
        assert isinstance(data["train_u"], PseudoLabeledDataset)
        # Verify __getitem__ works end-to-end (pseudo-target merged into item dict)
        item = data["train_u"][0]
        assert "target" in item
        assert item["target"].shape == (2,)
    finally:
        os.unlink(npz_path)
