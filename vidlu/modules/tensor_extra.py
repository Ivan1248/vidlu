from argparse import Namespace

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


def stop(x, k):
    y = x[:]
    if not has(x):
        return y
    y.extra = Namespace(**vars(x.extra))
    if has(y, k):
        delete(y, k)
    return y


class ExtraBase:
    name: str = None

    @classmethod
    def set(cls, x, value):
        set(x, cls.name, value)
        return x

    @classmethod
    def get(cls, x):
        return get(x, cls.name)

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
        x = next(vmu.extract_tensors(inputs))
        if cls.has(x):
            input_ladj = cls.get(x)
            ladj = lambda: closure() + input_ladj()
            # assert torch.all(ladj() >= 0),
            ladj()  # eager
            for yi in vmu.extract_tensors(outputs):
                cls.set(yi, ladj)
        return outputs
