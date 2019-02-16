import numpy as np

from vidlu.data.datasets import DatasetFactory
from scripts import dirs


class TestData:
    def test_datasets(self):
        factory = DatasetFactory(dirs.DATASETS)
        factory("whitenoise")
        for name in ['whitenoise', 'WhiteNoise', 'WHITENOISE', 'WHiTeNoiSe']:
            assert type(factory(name).all).__name__ == 'WhiteNoiseDataset'
        whitenoise = factory("whitenoise", size=42).all
        rademachernoise = factory("rademachernoise", size=9).all
        hblobs = factory("hblobs", size=42).all
        assert len(whitenoise) == 42
        assert len(rademachernoise) == 9
        assert len(hblobs) == 42
        assert all(np.all(a.x == b.x)
                   for a, b in zip(*[factory("hblobs", size=10, seed=6).all for _ in range(2)]))
        breakpoint()
        whitenoise.join(rademachernoise, hblobs)
