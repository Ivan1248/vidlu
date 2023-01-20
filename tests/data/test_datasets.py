from vidlu.data.datasets import DatasetFactory, Dataset


def test_datasets(tmpdir):
    factory = DatasetFactory(tmpdir)
    whitenoise = factory("WhiteNoise", size=53)
    rademachernoise = factory("RademacherNoise", size=9)
    hblobs = factory("HBlobs", size=42)
    assert isinstance(whitenoise, Dataset)
    assert isinstance(rademachernoise, Dataset)
    assert isinstance(hblobs, Dataset)
    assert len(whitenoise) == 53
    assert len(hblobs) == 42
    assert len(rademachernoise) == 9
    # assert len(hblobs) == 42
    # assert all(np.all(a.x == b.x)
    #           for a, b in zip(*[factory("hblobs", size=10, seed=6).all for _ in range(2)]))
    # whitenoise.join(rademachernoise, hblobs)
