from abc import ABC
from argparse import Namespace
import collections
import functools
from functools import reduce, partial
from typing import Union, Callable, Sequence, Mapping
import itertools
from os import PathLike
from fractions import Fraction
import re
import warnings

import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint

from vidlu.modules.utils import try_get_module_name_from_call_stack, sole_tuple_to_varargs
from vidlu.utils.collections import NameDict
from vidlu.utils.inspect import class_initializer_locals_c
from vidlu.utils.func import params, tryable, identity


# Module class extensions ##########################################################################

def _extended_interface(*superclasses):
    class ExtendedInterfaceModExt(*superclasses):
        def add_module(self, *args, **kwargs):
            if len(args) == 2 and len(kwargs) == 0:
                name, module = args
            elif len(args) == 0 and len(kwargs) == 1:
                name, module = next(iter(kwargs.items()))
            else:
                raise RuntimeError(
                    "Either 2 positional arguments or a single keyword argument is required.")
            super().add_module(name, module)

        def add_modules(self, *args, **kwargs):
            for name, module in dict(*args, **kwargs).items():
                super().add_module(name, module)

        @property
        def device(self):
            param = next(self.parameters(), None)
            return None if param is None else param[0].device

        def load_state_dict(self, state_dict_or_path, strict=True):
            """Handle a path being given instead of a file. (preferred since it
            automatically maps to the correct device). Taken from MagNet."""
            sd = state_dict_or_path
            if isinstance(sd, PathLike):
                sd = torch.load(sd, map_location=self.device)
            return super().load_state_dict(sd, strict=strict)

        def with_intermediate_outputs(self, submodule_paths: list):
            return with_intermediate_outputs(self, submodule_paths)

    return ExtendedInterfaceModExt


def _extract_tensors(*args, **kwargs):
    for a in itertools.chain(args, kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, Sequence):
            for x in _extract_tensors(*a):
                yield x
        elif isinstance(a, Mapping):
            for x in _extract_tensors(*a.values()):
                yield x


def _try_get_device_from_args(*args, **kwargs):
    x = next(_extract_tensors(*args, **kwargs), None)
    return None if x is None else x.device


def _buildable(*superclasses):
    class BuildableModExt(*superclasses):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            args = class_initializer_locals_c()
            self.args = NameDict(args)
            self._built = False

        def __call__(self, *args, **kwargs):
            try:
                if self._built:
                    return super().__call__(*args, **kwargs)
                else:
                    device = _try_get_device_from_args(*args, **kwargs)
                    if type(self).build != BuildableModExt.build:
                        self.build(*args, **kwargs)
                    if device is not None:
                        self.to(device)
                    if type(self).post_build != BuildableModExt.post_build:
                        super().__call__(*args, **kwargs)
                        self.post_build(*args, **kwargs)
                        if device is not None:
                            self.to(device)
                    self._built = True
                    return super().__call__(*args, **kwargs)
            except Exception as e:
                print(f"Error in {try_get_module_name_from_call_stack(self)}, {type(self)}")
                raise e

        def build(self, *args, **kwargs):
            pass

        def post_build(self, *args, **kwargs):
            """
            Leave this as it is if you don't need more initialization after
            evaluaton. I you do, make `build` return True.
            """
            pass

        def clone(self, **kwargs_override):
            arguments = dict(self.args)
            args = arguments.pop('args', ())
            kwargs = arguments.pop('kwargs', {})
            kwargs.update(**kwargs_override)
            return type(self)(*args, **arguments, **kwargs)

    return BuildableModExt


def _modifiable(*superclasses):
    class ModifibleModExt(*superclasses):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._modifiers = {}

        def set_modifiers(self, *args, **kwargs):
            self._modifiers.update(dict(*args, **kwargs))
            for k in [k for k, v in self._modifiers.items() if v is None]:
                del self._modifiers[k]
            return self

        def _call_with_modifier(self, name, *x):
            modifier = self._modifiers.get(name, None)
            module = self._modules[name]
            return (modifier(module) if modifier else module)(*x)

    return ModifibleModExt


def _stochastic(*superclasses):  # TODO: implement
    class StochasticModExt(*superclasses):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stoch_next_run_id = -1
            self._stoch_last_run_id = -1
            self._stoch_last_output = None
            # self._stochastic_scope = False

        def __call__(self, *args, stochastic_run_id=None, sample_count=None):
            if is_stochastic(self):
                self._stoch_last_run_id = self._stoch_last_run_id + 1
                return [
                    self.sample(*args, stochastic_run_id=self._stoch_last_run_id, sample_count=1)
                    for _ in sample_count]
                # if self._stochastic_scope and sample_count is None:
                #    raise ValueError("Stochastic scope modules require specifying sample_count.")
                # elif not self._stochastic_scope and sample_count is not None:
                #    raise ValueError(
                #        "sample_count needs to be None for non-stochastic-scope-modules.")
                # raise NotImplementedError("Stochastic modules need to override this method")
            if stochastic_run_id is None:
                stochastic_run_id = self._stoch_next_run_id
            if stochastic_run_id != self._stoch_last_run_id:
                self._stoch_last_output = super().__call__(*args)
            return self._stoch_last_output

        def sample(self, *args, stochastic_run_id=None, sample_count=None):
            return super().__call__(*args, stochastic_run_id=None, sample_count=None)

        def sample1(self, *args, sample_count=None):
            pass

        def is_stochastic(self):
            return is_stochastic(self)

        def stochastic_eval(self, sample_count=None):
            for c in self.children():
                c.stochastic_eval()

        def eval(self, sample_count=None, **kwargs):
            if sample_count is not None:
                self.stochastic_eval()
            super().eval(**kwargs)

    return StochasticModExt


def _potentially_invertible(*superclasses):
    class InvertibleModExt(*superclasses):
        @property
        def inverse(self):
            try:
                if hasattr(self, 'make_inverse'):
                    if not hasattr(self, '_inverse_cache'):
                        self._inverse_cache = (self.make_inverse(),)
                    return self._inverse_cache[0]
            except AttributeError as e:
                # Turn it into a TypeError so that it doesn't get turned into a confusing
                # AttributeError saying that this module has no `inverse` attribute
                raise TypeError(f"An inverse for the module `{type(self)}` is not defined: {e}")
            raise TypeError(f"An inverse for the module `{type(self)}` is not defined.")

        @inverse.setter
        def inverse(self, value):
            self._inverse_cache = (value,)

        def make_inverse(self):
            if hasattr(self, 'inverse_forward'):
                return Func(self.inverse_forward)
            raise TypeError(f"An inverse for the module `{type(self)}` is not defined.")

    return InvertibleModExt


def _extended(*superclasses):
    return [e(*superclasses) for e in [_buildable, _extended_interface, _potentially_invertible]]


# Core Modules #####################################################################################

class Module(*_extended(nn.Module, ABC), nn.Module):
    # Based on https://github.com/MagNet-DL/magnet/blob/master/magnet/nodes/nodes.py

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) + len(kwargs) > 0:
            self.args = NameDict(args=args, **kwargs)
        elif type(self).__init__ is not Module.__init__:
            self.args = NameDict(class_initializer_locals_c())
        self._built = False

    def __str__(self):
        return (self.__class__.__name__ + "("
                + ", ".join(f"{k}={v}" for k, v in self.args.items()) + ")")


class Identity(*_extended(nn.Identity)):
    def __init__(self):
        super().__init__(self)

    def make_inverse(self):
        return self


def _to_sequential_init_args(*args, **kwargs):
    if len(kwargs) > 0:
        if len(args) > 0:
            raise ValueError(
                "If keyword arguments are supplied, no positional arguments are allowed.")
        args = [kwargs]
    if len(args) == 1 and isinstance(args[0], dict):
        args = [collections.OrderedDict(args[0])]
    return args


class MultiModule(*_extended(nn.Sequential), _modifiable(nn.Sequential), nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*_to_sequential_init_args(*args, **kwargs))

    def __getitem__(self, idx):
        try:
            if isinstance(idx, slice) and any(isinstance(i, str) for i in [idx.start, idx.stop]):
                children_names = list(zip(*self.named_children()))[0]
                start, stop = idx.start, idx.stop
                if isinstance(start, str):
                    start = children_names.index(start)
                if isinstance(stop, str):
                    stop = children_names.index(stop)
                idx = slice(start, stop, idx.step)
        except ValueError:
            raise KeyError(f"Invalid index: {idx}.")
        if isinstance(idx, slice):
            return Seq(dict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return getattr(self, idx)
        return super().__getitem__(idx)

    def index(self, key):
        """Returns index of a child module from its name or the module itself."""
        return list(zip(*self.named_children()))[int(not isinstance(key, str))].index(key)

    def forward(self, *input):
        if len(self._modules) == 0 and len(input) != 1:
            raise RuntimeError("A `Seq` with no children can only accept 1 argument.")
        for name in self._modules:
            input = (self._call_with_modifier(name, *input),)
        return input[0]

    def make_inverse(self):
        result = Seq({k: m.inverse for k, m in reversed(self._modules.items())})
        result._modifiers = self._modifiers
        return result


class Seq(MultiModule):
    """
    A wrapper around torch.nn.Seq to enable passing a dict as the only
    parameter whereas in torch.nn.Seq only OrderedDict is accepted
    currently.
    It also supports slicing using strings.
    """

    def forward(self, *input):
        if len(self._modules) == 0 and len(input) != 1:
            raise RuntimeError("A `Seq` with no children can only accept 1 argument.")
        for name in self._modules:
            input = (self._call_with_modifier(name, *input),)
        return input[0]

    def make_inverse(self):
        result = Seq({k: m.inverse for k, m in reversed(self._modules.items())})
        result._modifiers = self._modifiers
        return result


# Fork, parallel, reduction, ... ###################################################################


class Fork(MultiModule):
    def forward(self, input):
        return tuple(m(input) for m in self)

    def make_inverse(self):
        return Fork({k.m.inverse() for k, m in self.named_children()})


class Parallel(MultiModule):
    def forward(self, *inputs):
        inputs = sole_tuple_to_varargs(inputs)
        if len(self) == 1:
            return [self[0](x) for x in inputs]
        elif len(inputs) != len(self):
            raise ValueError(f"The number of inputs ({len(inputs)}) does not"
                             + " match the number of parallel modules."
                             + f"\nError in {try_get_module_name_from_call_stack(self)}.")
        return tuple(m(x) for m, x in zip(self, inputs))


# class TuplePack(MultiModule):
#     def forward(self, input):
#         return (input,)
#
# class TupleUnpack(MultiModule):
#     def forward(self, input):
#         if not len(input)==1:
#             raise RuntimeError("The input should be a single-element tuple.")
#         return input[0]


class Merge(nn.Module):
    def forward(self, *inputs):
        inputs = sole_tuple_to_varargs(inputs)
        result = []
        for x in inputs:
            (result.extend if isinstance(x, tuple) else result.append)(x)
        return tuple(result)


class TupleSplit(nn.Module):
    def __init__(self, split_indices):
        super().__init__()

    def forward(self, x):
        result = []
        last_si = 0
        for si in self.args.split_indices:
            result.append(x[last_si:si])
            last_si = si
        result.append(x[self.args.split_indices[-1]:])
        return tuple(result)


class Reduce(Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    def forward(self, *inputs):
        inputs = sole_tuple_to_varargs(inputs)
        return reduce(self.func, inputs[1:], inputs[0].clone())


def pasum(x):
    def sum_pairs(l, r):
        return [a + b for a, b in zip(l, r)]

    def split(x):
        l, r = x[:len(x) // 2], x[len(x) // 2:]
        r, rem = r[:len(l)], r[len(l):]
        return x, r, rem

    while len(x) > 1:
        l, r, rem = split(x)
        x = sum_pairs(l, r) + rem

    return x[0]


class Sum(Module):  # TODO: rename to "Add"
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        inputs = sole_tuple_to_varargs(inputs)
        shape = inputs[0].shape
        for oo in inputs[1:]:
            if oo.shape != shape:
                print(try_get_module_name_from_call_stack(self),
                      ' '.join(str(tuple(x.shape)) for x in inputs))
        if len(inputs) == 1:
            return inputs[0]
        y = inputs[0] + inputs[1]
        for x in inputs[2:]:
            y += x
        return y


class Concat(Module):
    def __init__(self, dim=1):
        super().__init__()

    def forward(self, *inputs):
        inputs = sole_tuple_to_varargs(inputs)
        return torch.cat(inputs, self.args.dim)


class Split(Module):
    def __init__(self, split_size_or_sections: Union[int, Sequence], dim=1):
        super().__init__()

    def forward(self, x):
        return x.split(self.args.split_size_or_sections, dim=self.args.dim)

    def make_inverse(self):
        return Concat(self.args.dim)


class Chunk(Module):
    def __init__(self, chunk_count: int, dim=1):
        super().__init__()

    def forward(self, x):
        return x.chunk(self.args.chunk_count, dim=self.args.dim)

    def make_inverse(self):
        return Concat(self.args.dim)


class Permute(Module):
    def __init__(self, *dims):
        super().__init__()

    def forward(self, x):
        return x.permute(*self.args.dims)

    def make_inverse(self):
        dims = self.args.dims
        inv_dims = [-1] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        return Permute(*inv_dims)


class Transpose(Module):
    def __init__(self, dim0, dim1):
        super().__init__()

    def forward(self, x):
        return x.transpose(self.args.dim0, self.args.dim1)

    def make_inverse(self):
        return self


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()

    def forward(self, x):
        return torch.reshape(x, (x.shape[0], *self.args.shape))


class BatchReshape(Module):
    def __init__(self, *shape_or_func: Union[tuple, Callable[[tuple], tuple]]):
        if len(shape_or_func) == 1 and callable(shape_or_func[0]):
            shape_or_func = shape_or_func[0]
        super().__init__()

    def build(self, x):
        self.orig_shape = x.shape[1:]
        shape = self.args.shape_or_func
        self.shape = shape(*self.orig_shape) if callable(shape) else shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def inverse_forward(self, y):
        return y.reshape(y.shape[0], *self.orig_shape)


def _parse_auto_reshape_arg(dims_or_factors):
    dims_or_factors = re.findall(r" *(\([^)]*\)|[^(),]*) *(?:,|$)", dims_or_factors)
    return [-1 if x.strip() != '-1' else
            _parse_auto_reshape_arg(x[1:-1]) if x[0] == '(' else
            Fraction(x[1:].strip()) if x[0] == '*' else
            int(x) for x in dims_or_factors]


class AutoReshape(Module):
    """A reshape module that can be adaptive to input shape."""

    def __init__(self, dims_or_factors: Union[str, Sequence]):
        super().__init__()
        if isinstance(dims_or_factors, str):
            self.dims_or_factors = _parse_auto_reshape_arg(dims_or_factors)

    def build(self, x):
        def get_subshape(d, dims_or_factors):
            other = d // np.prod(f for f in dims_or_factors if f != -1)

            return sum([int(d * f) if isinstance(f, Fraction) else
                        int(d * (1 - other)) if f == -1 else
                        f for f in dims_or_factors], [])

        self.shape = [get_subshape(d, f) if isinstance(f, Sequence) else
                      [int(d * f) if isinstance(f, Fraction) else f] for d, f in
                      zip(x.shape, self.dims_or_factors)]

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(*self.shape)

    def make_inverse(self):
        return BatchReshape(*self.orig_shape)


class Contiguous(Module):
    def forward(self, x):
        return x.contiguous()

    def make_inverse(self):
        return Identity()


class ContiguousInv(Identity):
    def forward(self, x):
        return x

    def make_inverse(self):
        return Contiguous()


class Index(Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x.__getitem__(*self.args.args)


class To(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kwargs=kwargs)

    def __call__(self, x):
        return x.to(*self.args.args, **self.args.kwargs)


class Clamp(nn.Module):
    def __init__(self, min_=None, max_=None, inplace=False):
        self.min_, self.max_, self.inplace = min_, max_, inplace

    def forward(self, x):
        return torch.clamp(x, min=self.min_, max=self.max_, out=x if self.inplace else None)


# Debugging ########################################################################################

class Print(Identity):
    def __init__(self, func, text=""):
        super().__init__()
        self.func, self.text = func, text

    def forward(self, x):
        print(self.text, self.func(x))
        return x


class PrintAround(Module):
    def __init__(self, module, before=None, after=None):
        super().__init__()
        self.module, self.before, self.after = module, before, after

    def forward(self, x):
        if self.before:
            print(self.before(x))
        y = self.module(x)
        if self.after:
            print(self.after(y))
        return y


# Wrapped modules ##################################################################################

def _dimensional_build(name, input, args, in_channels_name='in_channels') -> nn.Module:
    if in_channels_name in args and args[in_channels_name] is None:
        args[in_channels_name] = input.shape[1]
    dim = len(input.shape) - 2  # assuming 1 batch and 1 channels dimension
    if dim not in [1, 2, 3]:
        raise ValueError(f"Cannot infer {name} dimension from input shape.")
    name = f"{name}{dim}d"
    layer_func = nn.__getattribute__(name)
    for k in params(layer_func).keys():
        if k not in args:
            raise ValueError(f"Missing argument for {name}: {k}.")
    module = layer_func(**args)
    return module


def _get_conv_padding(padding_type, kernel_size, dilation):
    if any(k % 2 == 0 for k in ([kernel_size] if isinstance(kernel_size, int) else kernel_size)):
        raise ValueError(f"`kernel_size` must be an odd positive integer "
                         f"or a sequence of them, not {kernel_size}.")
    if padding_type not in ('half', 'full'):
        raise ValueError(f"Invalid padding_type value {padding_type}.")

    def get_padding(k, d):
        return (k - 1) * d // 2 if padding_type == 'half' else (k - 1) * d

    if any(isinstance(x, Sequence) for x in [kernel_size, dilation]):
        if isinstance(dilation, int):
            dilation = [dilation] * len(kernel_size)
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(dilation)
        return tuple(get_padding(k, d) for k, d in zip(kernel_size, dilation))
    else:
        return get_padding(kernel_size, dilation)


class WrappedModule(Module):
    def __init__(self, orig=None):
        super().__init__()
        self.orig = orig

    def forward(self, x):
        return self.orig(x)

    def __repr__(self):
        return "A" + repr(self.orig)


# TODO: Make separate Conv*ds
class Conv(WrappedModule):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, in_channels=None, padding_mode='zeros'):
        super().__init__()
        self.args.padding = (_get_conv_padding(padding, kernel_size, dilation)
                             if isinstance(padding, str) else padding)

    def build(self, x):
        self.orig = _dimensional_build("Conv", x, self.args)


class MaxPool(WrappedModule):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        super().__init__()
        self.args.padding = (_get_conv_padding(padding, kernel_size, dilation)
                             if isinstance(padding, str) else padding)

    def build(self, x):
        self.orig = _dimensional_build("MaxPool", x, self.args)


class AvgPool(WrappedModule):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        self.args.padding = (_get_conv_padding(padding, kernel_size, dilation=1)
                             if isinstance(padding, str) else padding)

    def build(self, x):
        self.orig = _dimensional_build("AvgPool", x, self.args)


class ConvTranspose(WrappedModule):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=1, groups=1,
                 bias=True, dilation=1, in_channels=None):
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("ConvTranspose", x, self.args)


class Linear(WrappedModule):
    def __init__(self, out_features: int, bias=True, in_features=None):
        super().__init__()

    def build(self, x):
        self.args.in_features = self.args.in_features or np.prod(x.shape[1:])
        self.orig = nn.Linear(**{k: v for k, v in self.args.items()})

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.view(x.size(0), -1)
        return super().forward(x)


class BatchNorm(WrappedModule):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 num_features=None, support_checkpointing=False):
        self.support_checkpointing = support_checkpointing
        del support_checkpointing
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("BatchNorm", x, self.args, 'num_features')

    # This is probably commented for reproducibility of existing algorithms
    def forward(self, input):
        """Modified forward to make it work with checkpoints as it should.

        Based on
        https://github.com/csrhddlam/pytorch-checkpoint
        """

        if input.requires_grad or not self.support_checkpointing:
            if not input.requires_grad:
                warnings.warn("The default implementation of BatchNorm does not"
                              + " work correctly with checkpointing. Set"
                              + " `support_checkpointing=True` to fix it.")
            return super().forward(input)

        self.orig._check_input_dim(input)

        exponential_average_factor = 0.0

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)


class GhostBatchNorm(BatchNorm):
    # Based on https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
    def __init__(self, batch_size, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 num_features=None):
        super().__init__(eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats, num_features=num_features)
        self.batch_size = batch_size
        self.running_mean, self.running_var, self.num_splits = [None] * 3

    def build(self, x):
        self.orig = _dimensional_build("BatchNorm", x, self.args, 'num_features')
        num_splits = x.shape[0] // self.args.batch_size
        if num_splits * self.args.batch_size < x.shape[0]:
            raise RuntimeError(f"The size of tha input batch ({x.shape[0]}) must be divisible by"
                               + f" `batch_size` ({self.args.batch_size}).")
        self.register_buffer('running_mean', torch.zeros(self.orig.num_features * num_splits))
        self.register_buffer('running_var', torch.ones(self.orig.num_features * num_splits))
        self.running_mean = self.running_mean.view(num_splits, self.num_features).mean(
            dim=0).repeat(num_splits)
        self.running_var = self.running_var.view(num_splits, self.num_features).mean(dim=0).repeat(
            num_splits)
        self.num_splits = num_splits

    def forward(self, x):
        if self.training or not self.track_running_stats:
            _, C, *S = x.shape
            return F.batch_norm(
                x.view(-1, C * self.num_splits, *S), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(x.shape)
        else:
            return F.batch_norm(
                x, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


'''
# Wrap remaining torch.nn classes

_class_template = """
class {typename}(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.orig = nn.{typename}(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.orig(*args, **kwargs)
"""

_already_wrapped_classes = (
        list(list(zip(*inspect.getmembers(sys.M[__name__])))[0]) + ['Module'] +
        [f'Conv{t}{d}d' for t in ['', 'Transpose'] for d in {1, 2, 3}])

for name, cls in inspect.getmembers(nn, inspect.isclass):
    if issubclass(cls, nn.Module) and name not in _already_wrapped_classes:
        class_definition = _class_template.format(typename=name)
        print(class_definition)
        exec(class_definition)
'''


# Additional generally useful M ##############################################################


class _Func(Module):
    def __init__(self, func, func_inv=None):
        super().__init__()
        self._func = func

    def forward(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class Func(_Func):
    def __init__(self, func, func_inv=None):
        super().__init__(func)
        self._func_inv = func_inv

    def make_inverse(self):
        if self._inv is None:
            raise RuntimeError("Inverse not defined.")
        inv = Func(self._inv, self._func)
        inv.inverse = self
        return inv


# Stochastic #######################################################################################

class StochasticModule(Module, ABC):
    def __init__(self):
        super().__init__()
        self.stochastic_eval = False


def is_stochastic(module):
    return isinstance(module, StochasticModule) or any(is_stochastic(m) for m in module.children())


class _DropoutNd(StochasticModule, ABC):
    __constants__ = ['p', 'inplace']

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)


class Dropout(_DropoutNd):
    def forward(self, input):
        return F.dropout(input, self.p, training=self.training or self.stochastic_eval,
                         inplace=self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout2d(input, self.p, training=self.training or self.stochastic_eval,
                           inplace=self.inplace)


# Utilities ########################################################################################

class Adapter(Module):
    def __init__(self, module_or_factory, input_adapter=None, output_adapter=None):
        super().__init__()
        mof = module_or_factory
        self.module = mof if isinstance(mof, nn.Module) else mof()
        self.input_adapter = input_adapter or identity
        self.output_adapter = output_adapter or identity

    def forward(self, *args, **kwargs):
        return self.output_adapter(self.module(self.input_adapter(*args, **kwargs)))


def parameter_count(module) -> Namespace:
    trainable, non_trainable = 0, 0
    for _, p in module.named_parameters():
        n = np.prod(p.size())
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return Namespace(trainable=trainable, non_trainable=non_trainable)


def get_submodule(root_module, path: Union[str, Sequence]) -> Module:
    """
    Returns a submodule of `root_module` that corresponds to `path`. It works
    for other attributes (e.g. Parameters) too.
    Arguments:
        root_module (Module): a module.
        path (Tensor): a string with the name of the module relative to
            `root_module`.
    """
    if isinstance(path, str):
        path = path.split('.') if path != '' else []
    for name in path:
        if not hasattr(root_module, name):
            raise AttributeError(
                f"The '{type(root_module).__name__}' instance has no submodule '{name}'. It has"
                + f"  children: {', '.join(list(k for k, v in root_module.named_children()))}.")
        root_module = getattr(root_module, name)
    return root_module


def join_sequentials(a, b):
    return Seq(**{k: v for k, v in a.named_children()}, **{k: v for k, v in b.named_children()})


def deep_split(root: nn.Module, split_path: Union[list, str]):
    if isinstance(split_path, str):
        split_path = [] if split_path == '' else split_path.split('.')
    if len(split_path) == 0:
        return root, Seq()
    next_name, path_remainder = split_path[0], split_path[1:]
    if isinstance(root, Seq):
        split_index = root.index(next_name)
        left, right = root[:split_index + 1], root[split_index + int(len(path_remainder) == 0):]
        if len(path_remainder) > 0:
            left[-1] = deep_split(left[-1], path_remainder)[0]
            right[0] = deep_split(right[0], path_remainder)[1]
        return left, right
    else:
        raise NotImplementedError(f"Splitting not implemented for module type {type(root)}")


def deep_join(left: nn.Module, right: nn.Module):
    if not type(left) is type(right):
        raise ValueError("Both modules must be of the same type.")
    if isinstance(left, Seq):
        def index_to_name(module, index):
            return list(module.named_children())[index][0]

        if len(left) * len(right) == 0 or index_to_name(left, -1) != index_to_name(right, 0):
            return join_sequentials(left, right)
        left = left[:]
        left[-1] = deep_join(left[-1], right[0])
        return join_sequentials(left, right[1:])
    else:
        raise NotImplementedError(f"Joining not implemented for module type {type(left)}")


def with_intermediate_outputs(root: nn.Module, submodule_paths: list):
    """Creates a function extending `root.forward` so that a pair
    containing the output of `root.forward` as well as well as a list of
    intermediate outputs as defined in `submodule_paths`.

    Arguments:
        root (Module): a module.
        submodule_paths (List[str]): a list of names (relative to `root`)
        of modules the outputs of which you want to get.

    Example:
        >>> module(x)
        tensor(...)
        >>> module_wio = with_intermediate_outputs(module, 'backbone', 'head.logits')
        >>> module_wio(x)
        tensor(...), (tensor(...), tensor(...))
    """
    if isinstance(submodule_paths, str):
        submodule_paths = [submodule_paths]

    def get_submodules():
        return [get_submodule(root, p) for p in submodule_paths]

    @functools.wraps(root)
    def wrapper(*args, **kwargs):
        submodules = tryable(get_submodules, None)()
        if submodules is None:  # in case the module is not yet built
            root(*args, **kwargs)
            submodules = get_submodules()

        outputs = [None] * len(submodule_paths)

        def create_hook(idx):
            def hook(module, input, output):
                outputs[idx] = output

            return hook

        handles = [m.register_forward_hook(create_hook(i)) for i, m in enumerate(submodules)]
        output = root(*args, **kwargs)
        for h in handles:
            h.remove()

        return output, outputs

    return wrapper


class IntermediateOutputsModuleWrapper(Module):
    def __init__(self, module, submodule_paths):
        """
        Creates a function extending `root.forward` so that a pair containing
        th output of `root.forward` as well as well as a list of intermediate
        outputs as defined in `submodule_paths`.
        Arguments:
            root (Module): a module.
            submodule_paths (List[str]): a list of module names relative to
            `root`.
        """
        super().__init__()
        self.module = module
        self.handles, self.outputs = None, None

    def __del__(self):
        if self.handles is not None:
            for h in self.handles:
                h.remove()

    def post_build(self, *args, **kwargs):
        def create_hook(idx):
            def hook(module, input, output):
                self.outputs[idx] = output

            return hook

        submodules = [get_submodule(self.module, p) for p in self.args.submodule_paths]
        self.handles = [m.register_forward_hook(create_hook(i))
                        for i, m in enumerate(submodules)]

    def forward(self, *args, **kwargs):
        self.outputs = [None] * len(self.args.submodule_paths)
        output = self.module(*args, **kwargs)
        outputs = self.outputs
        self.outputs = None
        return output, tuple(outputs)


class CheckpointingModuleWrapper(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return checkpoint(self.module, *args)


def checkpointed(module):
    return partial(checkpoint, module)


# Gradiont modification

class _RevGrad(torch.autograd.Function):
    """From https://github.com/janfreyberg/pytorch-revgrad"""

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


rev_grad = _RevGrad.apply


class RevGrad(Module):
    def forward(self, x):
        return rev_grad(x)


class _AmpGrad(torch.autograd.Function):
    def __init__(self, a):
        self.a = a

    def forward(self, input_):
        self.save_for_backward(input_)
        output = input_
        return output

    def backward(self, grad_output):  # pragma: no cover
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = self.a * grad_output
        return grad_input


amp_grad = _AmpGrad.apply


class AmpGrad(Module):
    def forward(self, x):
        return amp_grad(x)


class StopGrad(Module):
    def forward(self, x):
        return x.detach()
