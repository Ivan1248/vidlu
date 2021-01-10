from argparse import Namespace
from vidlu.utils.func import partial

import torch

import vidlu.modules.utils as vmu
from vidlu.utils.func import Cached


class _NoValue:
    pass


def has(x, k=None):
    return hasattr(x, 'extra') and (k is None or hasattr(x.extra, k))


def get(x, k, default=_NoValue):
    if default is not _NoValue and not hasattr(x, 'extra') or not hasattr(x.extra, k):
        return default
    return getattr(x.extra, k)


def set(x, k, value):
    if not has(x):
        x.extra = Namespace()
    setattr(x.extra, k, value)


def delete(x, k):
    delattr(x.extra, k)


def clear(x):
    del x.extra


def pop(x, k):
    v = get(x, k)
    delete(x, k)
    return v


def _stop(x, k, must_exist=False):
    y = x[:]
    if not has(x):
        return y
    y.extra = Namespace(**vars(x.extra))
    if has(y, k) or must_exist:
        delete(y, k)
    return y


def stop(x, k, must_exist=False):
    return vmu.map_tensors(x, partial(_stop, k=k, must_exist=must_exist))


class ExtraBase:
    name: str = None

    @classmethod
    def set(cls, x, value):
        set(x, cls.name, value)
        return x

    @classmethod
    def get(cls, x, default=_NoValue):
        return get(x, cls.name, default=default)

    @classmethod
    def has(cls, x):
        return has(x, cls.name)

    @classmethod
    def pop(cls, x):
        return pop(x, cls.name)

    @classmethod
    def stop(cls, x):
        return stop(x, cls.name)


class LogAbsDetJac(ExtraBase):
    name = 'ladj'

    @classmethod
    def set(cls, x, value):
        if not callable(value):
            value_ = value
            value = lambda: value_
        return super().set(x, Cached(value))

    @classmethod
    def zero(cls, x):
        return lambda: torch.zeros(len(x), dtype=x.dtype, device=x.device)

    @classmethod
    def add(cls, outputs, inputs, closure):
        if not callable(closure):
            raise TypeError("The closure must be callable.")
        x = next(vmu.extract_tensors(inputs))
        if cls.has(x):
            input_ladj = cls.get(x)
            ladj = lambda: closure() + input_ladj()
            # assert torch.all(ladj() >= 0),
            ladj()  # eager
            for yi in vmu.extract_tensors(outputs):
                cls.set(yi, ladj)
        return outputs


class Name(ExtraBase):
    name = 'name'

    @classmethod
    def set(cls, x, name):
        return super().set(x, name)

    @classmethod
    def add(cls, x, name):
        for xi in vmu.extract_tensors(x):
            if not cls.has(x):
                cls.set(xi, name)
            else:
                prev_name = cls.get(x)
                from os.path import commonprefix
                prefix = commonprefix([prev_name, name])
                p = len(prefix)
                try:
                    cls.set(xi,
                            f"{prefix}({prev_name[p:]}, {name[p:]})" if p > 0 and prefix[-1] != '(' else
                            f"{prefix}{prev_name[p:-1]}, {name[p:]})")
                except:
                    breakpoint()
        return x
