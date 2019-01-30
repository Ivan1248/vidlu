import pytest
from functools import partial


class TestHardPartial:
    def test_hard_partial(self):
        from vidlu.utils.func import hard_partial

        def carry(a, coconut=4):
            return a * coconut ** 2

        f1 = hard_partial(carry, 2)
        f2 = hard_partial(carry, a=2)
        assert f1.func is f2.func
        assert f1.args == (2,)
        assert f2.args == ()
        assert len(f1.keywords) == 0
        assert f2.keywords == dict(a=2)
        assert f2(coconut=1) == 2
        with pytest.raises(Exception):
            f2(a=5)

        f4 = hard_partial(f2, coconut=6)
        assert f4.func == f2
        assert f4.keywords == dict(coconut=6)
        with pytest.raises(Exception):
            f4(coconut=1)

        f5 = partial(carry, a=7, coconut=8)
        f6 = hard_partial(f5)
        assert len(f6.args) == 0
        assert len(f6.keywords) == 0
        assert f6.func == f5
        f6(a=3, coconut=2)

        f5 = hard_partial(carry, a=7, coconut=8)
        f6 = partial(f5)
        assert len(f6.args) == 0
        assert len(f6.keywords) == 0
        assert f6.func == f5
        f6()
        with pytest.raises(Exception):
            f6(a=3)
        with pytest.raises(Exception):
            f6(coconut=3)
