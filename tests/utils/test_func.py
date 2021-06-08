from dataclasses import dataclass

import pytest
from functools import partial, wraps
from vidlu.utils.func import (Empty, default_args, params, func_to_class, class_to_func)
from vidlu.utils.func import FuncTree as ft, IndexableUpdatree as it, StrictObjectUpdatree as ot


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
        from vidlu.utils.func import frozen_partial

        def carry(a, coconut=4):
            return a * coconut ** 2

        f1 = frozen_partial(carry, 2)
        f2 = frozen_partial(carry, a=2)
        assert f1.func is f2.func
        assert f1.args == (2,)
        assert f2.args == ()
        assert len(f1.keywords) == 0
        assert f2.keywords == dict(a=2)
        assert f2(coconut=1) == 2
        with pytest.raises(Exception):
            f2(a=5)

        f4 = frozen_partial(f2, coconut=6)
        assert f4.func == f2
        assert f4.keywords == dict(coconut=6)
        with pytest.raises(Exception):
            f4(coconut=1)

        f5 = partial(carry, a=7, coconut=8)
        f6 = frozen_partial(f5)
        assert len(f6.args) == 0
        assert len(f6.keywords) == 0
        assert f6.func == f5
        f6(a=3, coconut=2)

        f5 = frozen_partial(carry, a=7, coconut=8)
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


class TestFuncToClassAndClassToFunc:
    def test_func_to_class(self):
        def carry_thing(x, destination, swallow_type='african'):
            return x, destination, swallow_type

        CarryThing = func_to_class(carry_thing)
        assert CarryThing.__name__ == "CarryThing"
        assert CarryThing.__module__ == carry_thing.__module__

        assert CarryThing(destination='Goat')('coconut') == carry_thing('coconut', 'Goat')

        CarryThing2 = func_to_class(carry_thing, 2, name="CarryThing2")

        assert (CarryThing2('European')('holy hand grenade', 'Caerbannog')
                == carry_thing('holy hand grenade', 'Caerbannog', 'European'))

    def test_class_to_func(self):
        @dataclass
        class CarryThing:
            destination: str
            swallow_type: str = 'african'

            def __call__(self, x):
                return x, self.destination, self.swallow_type

        carry_thing = class_to_func(CarryThing)
        assert carry_thing.__name__ == "carry_thing"
        assert carry_thing.__module__ == CarryThing.__module__

        assert carry_thing('coconut', 'Antioch') == CarryThing('Antioch')('coconut')

    def test_func_to_class_to_func_to_class(self):
        def carry_thing1(x, destination, swallow_type='african'):
            return x, destination, swallow_type

        CarryThing1 = func_to_class(carry_thing1)
        carry_thing2 = class_to_func(CarryThing1)
        CarryThing2 = func_to_class(carry_thing2)

        assert (carry_thing1('coconut', 'Antioch')
                == carry_thing2('coconut', 'Antioch')
                == CarryThing1('Antioch')('coconut')
                == CarryThing2('Antioch')('coconut'))

    def test_func_to_class_wraps(self):
        def carry_thing(x, destination, swallow_type='african'):
            return x, destination, swallow_type

        @wraps(carry_thing)
        def carry_thing_wrapper(x, *a, **k):
            return carry_thing(x, *a, **k)

        CarryThing = func_to_class(carry_thing_wrapper)

        assert CarryThing(destination='Caerbannog')('rabbit') == carry_thing('rabbit', 'Caerbannog')


class TestUpdatrees:
    def test_updatrees(self):
        import vidlu.configs.training as vct

        argtree = ot(attack_f=ft(loss=vct.losses.crossentropy_ll,
                                 initializer=ot(tps=ot(name_to_mean_std=it(offsets=(0, 0.05))))),
                     train_step=ot(mem_efficient=False))

        conf = argtree.apply(vct.semisup_cons_phtps20)
        assert params(conf.attack_f).loss == vct.losses.crossentropy_ll
        assert params(conf.attack_f).initializer.tps.name_to_mean_std['offsets'] == (0, 0.05)
        assert not conf.train_step.mem_efficient

        argtree_nea = ot(argtree, nonexisting_attr=None)
        with pytest.raises(AttributeError):
            argtree_nea.apply(vct.semisup_cons_phtps20)
