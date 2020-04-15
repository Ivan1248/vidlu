from vidlu.data.datasets import DatasetFactory


def test_datasets(tmpdir):
    factory = DatasetFactory(tmpdir)
    factory("whitenoise")
    for name in ['whitenoise', 'WhiteNoise', 'WHITENOISE', 'WHiTeNoiSe']:
        assert type(factory(name).all).__name__ == 'WhiteNoise'
    whitenoise = factory("whitenoise", size=42).all
    rademachernoise = factory("rademachernoise", size=9).all
    # TODO: uncomment hblobs
    # hblobs = factory("hblobs", size=42).all
    assert len(whitenoise) == 42
    assert len(rademachernoise) == 9
    # assert len(hblobs) == 42
    # assert all(np.all(a.x == b.x)
    #           for a, b in zip(*[factory("hblobs", size=10, seed=6).all for _ in range(2)]))
    # whitenoise.join(rademachernoise, hblobs)
