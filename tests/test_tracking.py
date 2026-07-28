import numpy as np

from vidlu.tracking import flatten_scalars


class TestFlattenScalars:
    def test_scalars_kept(self):
        assert flatten_scalars(dict(a=1, b=2.5, c=True)) == dict(a=1, b=2.5, c=True)

    def test_numpy_scalars_converted(self):
        result = flatten_scalars(dict(f=np.float64(0.5), i=np.int32(3), z=np.array(2.0)))
        assert result == dict(f=0.5, i=3, z=2.0)
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_nested_dicts_flattened_to_dotted_keys(self):
        result = flatten_scalars(dict(A=dict(P=0.1, R=dict(x=0.2)), b=3))
        assert result == {"A.P": 0.1, "A.R.x": 0.2, "b": 3}

    def test_non_scalars_dropped(self):
        result = flatten_scalars(dict(cm=np.eye(2), v=np.arange(3), s="text", a=1))
        assert result == dict(a=1)

    def test_per_attribute_dict_flattened(self):
        # Per-attribute metrics arrive as {metric: {attr_name: scalar}} and must
        # be flattened to dotted keys so each attribute becomes its own series.
        result = flatten_scalars({"mF1": {"sidewalk": 0.7, "surface": 0.8}, "amF1": 0.75})
        assert result == {"mF1.sidewalk": 0.7, "mF1.surface": 0.8, "amF1": 0.75}


def test_wandb_tracker_step_clamp():
    from vidlu.tracking import WandbTracker
    tracker = object.__new__(WandbTracker)  # bypass __init__ to avoid needing wandb
    tracker._last_step = 0
    assert tracker._clamp_step(5) == 5
    assert tracker._clamp_step(3) == 5  # non-monotonic input clamped
    assert tracker._clamp_step(7) == 7
