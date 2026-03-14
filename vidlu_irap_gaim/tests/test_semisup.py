import torch
from unittest.mock import patch

from vidlu_irap_gaim.training.semisup import multi_attribute_kl_div_ll, make_semisup_bih_data
from vidlu.data import Dataset


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


def test_make_semisup_bih_data_splitting():
    # Mock make_bih_data to return a dummy dataset
    mock_ds = {"train": MockDataset(100), "val": MockDataset(10), "test": MockDataset(10)}

    with patch("vidlu_irap_gaim.semisup.make_bih_data", return_value=mock_ds):
        # 10% labeled -> 10 labeled, 90 unlabeled
        data = make_semisup_bih_data(labeled_ratio=0.1, seed=42)

        # data["train"] is the labeled dataset, data["train_u"] is the unlabeled dataset
        train_l = data["train"]
        train_u = data["train_u"]
        assert len(train_l) == 10
        assert len(train_u) == 90

        # Verify disjointness (mock data is indices 0-99)
        # We need to access the underlying data or indices.
        # Since permute/split wraps in SubDataset ops, we can just check if we can iterate
        l_indices = set(list(train_l))
        u_indices = set(list(train_u))

        assert len(l_indices.intersection(u_indices)) == 0
        assert len(l_indices) + len(u_indices) == 100


def test_make_semisup_bih_data_no_shuffle():
    # Test that shuffle=False takes the first portion without shuffling
    mock_ds = {"train": MockDataset(100), "val": MockDataset(10), "test": MockDataset(10)}

    with patch("vidlu_irap_gaim.semisup.make_bih_data", return_value=mock_ds):
        # 10% labeled without shuffling -> first 10 samples
        data = make_semisup_bih_data(labeled_ratio=0.1, shuffle=False)

        train_l = data["train"]
        train_u = data["train_u"]
        assert len(train_l) == 10
        assert len(train_u) == 90

        # Without shuffling, labeled should be first 10 indices (0-9)
        l_indices = list(train_l)
        assert l_indices == list(range(10))
