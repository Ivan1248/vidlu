import pytest
from functools import partial
from vidlu.utils.func import Empty, default_args, params


class TestHardPartial:
    def test_params(self):
        def foo(a, b, c=8):
            pass

        p = params(foo)
        assert len(p) == 3 and dict(**p) == dict(a=Empty, b=Empty, c=8)

    def test_params_empty(self):
        def foo(a=Empty, b=Empty):
            pass

        def bar(a, b):
            pass

        assert params(foo) == params(bar)

    def test_default_args(self):
        def foo(a, b, c=8):
            pass

        da = default_args(foo)
        assert len(da) == 1 and da['c'] == 8

    def test_default_args_empty(self):
        def foo(a=Empty, b=Empty):
            pass

        assert len(default_args(foo)) == 0

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

    def test_inherit_missing_args(self):
        from vidlu.utils.func import inherit_missing_args, Empty

        def foo(a=1, b=2):
            return a, b

        inherit_foo_args = inherit_missing_args(foo)

        @inherit_foo_args
        def unladen_swallow(a, b):
            return a, b

        assert unladen_swallow() == (1, 2)
        assert inherit_foo_args(lambda a, b: (a, b))() == (1, 2)
        assert inherit_foo_args(lambda a=Empty, b=Empty: (a, b))() == (1, 2)
        assert inherit_foo_args(lambda a, b=4: (a, b))() == (1, 4)
        assert inherit_foo_args(lambda a=3, b=Empty: (a, b))() == (3, 2)
        assert inherit_foo_args(lambda a=3, b=4: (a, b))() == (3, 4)

        for bar in [lambda a, b=2: (a, b), lambda a=Empty, b=2: (a, b)]:
            bar_decorator = inherit_missing_args(bar)

            assert bar_decorator(lambda a, b: (a, b))(a=1) == (1, 2)
            assert bar_decorator(lambda a, b: (a, b))(1) == (1, 2)
            with pytest.raises(TypeError):
                assert bar_decorator(lambda a, b: (a, b))()
