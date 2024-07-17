import pytest

import random
from vidlu import factories


def test_get_data_single(tmpdir):
    datasets, keys, transform_str = factories.get_data("DummyDataset{all,all}", tmpdir)
    assert keys == [("DummyDataset", "all")] * 2


def test_get_data_single_args(tmpdir):
    example_shape = (random.randint(1, 32),) * 2 + (3,)
    datasets, keys, transform_str = factories.get_data(
        f"WhiteNoise(example_shape={example_shape},key='x')", tmpdir)
    assert keys[0] == f"WhiteNoise(example_shape={example_shape},key='x')"
    assert datasets[0][0].x.shape == example_shape


def test_get_data_multiple(tmpdir):
    datasets, keys, transform_str = factories.get_data(
        "DummyDataset{train,val}, WhiteNoise(example_shape=(8,8,8)): (d[0],d[1][:11],d[2])",
        tmpdir)
    assert len(datasets) == 3
    assert len(datasets[1]) == 11
