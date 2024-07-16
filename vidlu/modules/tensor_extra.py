from argparse import Namespace
from vidlu.utils.func import partial

import torch

import vidlu.modules.utils as vmu
from vidlu.utils.func import Cached


class _NoValue:
    pass


def has(x: torch.Tensor, k:str=None):
    return hasattr(x, 'extra') and (k is None or hasattr(x.extra, k))


def get(x: torch.Tensor, k:str, default=_NoValue):
    if default is not _NoValue and not hasattr(x, 'extra') or not hasattr(x.extra, k):
        return default
    return getattr(x.extra, k)


def set(x: torch.Tensor, k:str, value):
    if not has(x):
        x.extra = Namespace()
    setattr(x.extra, k, value)


def delete(x: torch.Tensor, k:str):
    delattr(x.extra, k)


def clear(x):
    del x.extra


def pop(x: torch.Tensor, k:str):
    v = get(x, k)
    delete(x, k)
    return v


def _stop(x: torch.Tensor, k:str, must_exist=False):
    y = x[:]
    if not has(x):
        return y
    y.extra = Namespace(**vars(x.extra))
    if has(y, k) or must_exist:
        delete(y, k)
    return y


def stop(x: torch.Tensor, k:str, must_exist=False):
    return vmu.map_tensors(x, partial(_stop, k=k, must_exist=must_exist))


class ExtraBase:
    name: str = None

    @classmethod
    def set(cls, x, value):
        """Sets the value of the attribute of `x.extra` with name `cls.name`."""
        set(x, cls.name, value)
        return x

    @classmethod
    def get(cls, x, default=_NoValue):
        """Gets the value of the attribute of `x.extra` with name `cls.name`."""
        return get(x, cls.name, default=default)

    @classmethod
    def has(cls, x):
        """Sets the value of the attribute of `x.extra` with name `cls.name`."""
        return has(x, cls.name)

    @classmethod
    def pop(cls, x):
        """Pops the value of the attribute of `x.extra` with name `cls.name`."""
        return pop(x, cls.name)

    @classmethod
    def stop(cls, x):
        """Returns an equivalent tensor with the attribute of `x.extra` with name `cls.name`
        removed.
        """
        return stop(x, cls.name)


class LogAbsDetJac(ExtraBase):  # TODO: rename to LogVolChange?
    """A class for manipulating the closure for computing the natural logarithm of the absolute
    Jacobian of the determinant of the output with respect to the input. The updated closure is to
    be stored in the metadata of the output tensors of invertible operatorions."""

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
                            f"{prefix}({prev_name[p:]}, {name[p:]})"
                            if p > 0 and prefix[-1] != '(' else
                            f"{prefix}{prev_name[p:-1]}, {name[p:]})")
                except:
                    breakpoint()
        return x
