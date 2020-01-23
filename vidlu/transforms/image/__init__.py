import torch

from vidlu.utils.func import func_to_class, class_to_func, make_multiinput

from .color import *
from .format import *
from .space import *
from .misc import *


# Elementwise ######################################################################################

@make_multiinput
def mul(x, factor):
    return x * factor


Mul = func_to_class(mul)


@make_multiinput
def div(x, divisor):
    return x / divisor


Div = func_to_class(div)


class Standardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return (x - self.mean.to(**fmt)) / self.std.to(**fmt)


standardize = class_to_func(Standardize)


class Destandardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return x * self.std.to(**fmt) + self.mean.to(**fmt)


destandardize = class_to_func(Destandardize)


class ClipBounds:
    def __init__(self, min_, max_):
        min_, max_ = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in [min_, max_]]
        self.min, self.max = [x.view(-1, 1, 1) for x in [min_, max_]]

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return x * self.std.to(**fmt) + self.mean.to(**fmt)


clip_bounds = class_to_func(ClipBounds)

_this = (lambda: None).__module__
__all__ = [k for k, v in locals().items()
           if hasattr(v, '__module__') and v.__module__ == _this and not k[0] == '_']
