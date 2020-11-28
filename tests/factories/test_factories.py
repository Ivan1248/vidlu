import pytest

import random
from vidlu import factories
from vidlu.utils import tree


def test_get_data_single(tmpdir):
    data = dict(factories.get_data("WhiteNoise{trainval,test}", tmpdir))
    assert list(data.keys()) == [("WhiteNoise", "trainval"), ("WhiteNoise", "test")]


def test_get_datasets_single_args(tmpdir):
    example_shape = (random.randint(1, 32),) * 2 + (3,)
    data = dict(factories.get_data(f"WhiteNoise(example_shape={example_shape}){{train,val}}", tmpdir))
    assert list(data.keys())[0][0] == (f"WhiteNoise(example_shape={example_shape})")
    train, val = dict(tree.flatten(data)).values()
    assert train[0].x.shape == val[0].x.shape == example_shape


def test_get_datasets_multiple(tmpdir):
    data = dict(factories.get_data("WhiteNoise{trainval,test}, WhiteNoise(example_shape=(8,8,8)){val}",
                                   tmpdir))
    assert len(data) == 3
