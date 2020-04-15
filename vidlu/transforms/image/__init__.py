import vidlu.utils.func as vuf
from .format import *
from .space import *
from .misc import *


# Elementwise ######################################################################################

@vuf.make_multiinput
def mul(x, factor):
    return x * factor


Mul = vuf.func_to_class(mul)


@vuf.make_multiinput
def div(x, divisor):
    return x / divisor


Div = vuf.func_to_class(div)


class Standardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return (x - self.mean.to(**fmt)) / self.std.to(**fmt)


standardize = vuf.class_to_func(Standardize)


class Destandardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return x * self.std.to(**fmt) + self.mean.to(**fmt)


destandardize = vuf.class_to_func(Destandardize)

_this = (lambda: None).__module__
__all__ = [k for k, v in locals().items()
           if hasattr(v, '__module__') and v.__module__ == _this and not k[0] == '_']
