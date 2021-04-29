import pytest

import random
from vidlu import factories
from vidlu.utils import tree


def test_get_data_single(tmpdir):
    data = factories.get_data("WhiteNoise{all,all}", tmpdir)
    assert [k for k, _ in data] == [("WhiteNoise", "all")] * 2


def test_get_data_single_args(tmpdir):
    example_shape = (random.randint(1, 32),) * 2 + (3,)
    data = factories.get_data(f"WhiteNoise(example_shape={example_shape}){{all}}", tmpdir)
    assert data[0][0] == (f"WhiteNoise(example_shape={example_shape})", 'all')
    ds = data[0][1]
    assert ds[0].x.shape == example_shape


def test_get_data_multiple(tmpdir):
    data = factories.get_data("WhiteNoise{all,all}, WhiteNoise(example_shape=(8,8,8)){all}",
                              tmpdir)
    assert len(data) == 3
