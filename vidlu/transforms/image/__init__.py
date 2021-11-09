import dataclasses as dc

import vidlu.utils.func as vuf
from .format import *
from .space import *
from .misc import *


# Elementwise ######################################################################################

@vuf.vectorize
def mul(x, factor):
    return x * factor


Mul = vuf.func_to_class(mul)


@vuf.vectorize
def div(x, divisor):
    return x / divisor


Div = vuf.func_to_class(div)


@dc.dataclass
class Standardize:
    mean: torch.Tensor
    std: torch.Tensor

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return (x - self.mean.view(-1, 1, 1).to(**fmt)) / self.std.view(-1, 1, 1).to(**fmt)


standardize = vuf.class_to_func(Standardize)


@dc.dataclass
class Destandardize:
    mean: torch.Tensor
    std: torch.Tensor

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return x * self.std.view(-1, 1, 1).to(**fmt) + self.mean.view(-1, 1, 1).to(**fmt)


destandardize = vuf.class_to_func(Destandardize)

_this = (lambda: None).__module__
__all__ = [k for k, v in locals().items()
           if hasattr(v, '__module__') and v.__module__ == _this and not k[0] == '_']
