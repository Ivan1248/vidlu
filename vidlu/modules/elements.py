from abc import ABC
from argparse import Namespace
import collections
import functools
from functools import reduce, partial
import typing as T
import itertools
from os import PathLike
from fractions import Fraction
import re
import warnings
import inspect
import copy
import weakref

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules as M
import torch.utils.checkpoint as cp
import einops

from vidlu.utils.collections import NameDict
from vidlu.utils.inspect import class_initializer_locals_c
import vidlu.utils.func as vuf
import vidlu.torch_utils as vtu
from vidlu.utils import tree

import vidlu.modules.utils as vmu
from vidlu.modules.deconv import FastDeconv

# Some of modules and functions from torch.nn are replaced with wrappers.
# Look for references of the `replaces` procedure to find the code doing it.

_replaces = []


def replaces(*names):
    for name in names:
        _replaces.append(name)
    return lambda x: x


# Module class extensions ##########################################################################


def _extract_tensors(*args, **kwargs):
    for a in itertools.chain(args, kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, T.Sequence):
            for x in _extract_tensors(*a):
                yield x
        elif isinstance(a, T.Mapping):
            for x in _extract_tensors(*a.values()):
                yield x


def _try_get_device_from_args(*args, **kwargs):
    x = next(_extract_tensors(*args, **kwargs), None)
    return None if x is None else x.device


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


class SplittableMixin:
    def split(self, submodule_name):
        raise TypeError(f"Splitting not implemented for module type {type(self)}")

    def deep_split(self, submodule_path):
        raise TypeError(
            f"Deep splitting not implemented for module type {type(self)}")

    def join(self, other):
        raise TypeError(f"Joining not implemented for module type {type(self)}")


def module_string(module, format='name (type)'):
    if format == 'name (type)':
        return f"{vmu.try_get_module_name_from_call_stack(module)} ({type(module).__name__})"
    else:
        raise NotImplementedError()


def register_inverse_check_hook(module):
    def check_inverse(module, input, output):
        # `handle.remove()` must come before calling `module.inverse` to avoid this being called
        # again under the `module.inverse` call, e.g. when `module.inverse` is `module` itself.
        handle.remove()
        rec = module.inverse(output)
        inp_list, rec_list = (list(_extract_tensors(x)) for x in [input, rec])

        if (ninp := len(inp_list)) != (nrec := len(rec_list)):
            raise RuntimeError(
                f"The number of inputs ({ninp}) does not match the number of reconstucted inputs"
                + f" {nrec} for {module_string(module)}")
        for inp, rec in zip(inp_list, rec_list):
            if not torch.allclose(inp, rec, **module.reconstruction_tols):
                adiff = (inp - rec).abs_()
                rdiff = adiff / (torch.min(inp.abs(), rec.abs()) + 1e-8)
                amax, amean, rmax, rmean = adiff.max(), adiff.mean(), rdiff.max(), rdiff.mean()
                raise RuntimeError(
                    f"Reconstructed inputs are not close enough to original inputs for"
                    + f" {module_string(module)}."
                    + f" Difference: {amax=:.2e}, {amean=:.2e}, {rmax=:.2e}, {rmean=:.2e}.")

    handle = module.register_forward_hook(check_inverse)
    return handle


class InverseError(RuntimeError):
    pass


class InvertibleMixin:
    # Note: reconstruction_tols is used in inverse check.
    # It can be overridden with a property.
    reconstruction_tols = dict(rtol=10, atol=1e-5)

    def _del_inverse_cache(self):
        del self.inverse

    @functools.cached_property
    def inverse(self: nn.Module) -> nn.Module:
        try:
            module = self.inverse_module()
            # The inverse of the inverse is the original module. `property` is used so that the
            # original module is not recognized as a child of the inverse module as this would
            # result in an infinite recursion when the `children` method is called on the inverse.
            module.inverse = property(weakref.ref(self, lambda s: module._del_inverse_cache))
            return module
        except TypeError as e:
            # Turn it into a TypeError so that it doesn't get turned into a confusing
            # AttributeError saying that this module has no `inverse` attribute
            raise TypeError(f"The inverse for the module `{type(self)}` is not defined: {e}")

    def inverse_module(self) -> nn.Module:
        if type(self).inverse_forward is not InvertibleMixin.inverse_forward:
            return Inverse(self)
        raise TypeError(f"`inverse_forward` is not defined for `{type(self)}`."
                        + " Either `inverse_forward` or `inverse_module` should be overloaded.")

    def inverse_forward(*args, **kwargs):
        # overload either this or inverse_module
        raise InverseError()

    @functools.cached_property
    def flow(self) -> nn.Module:
        return self.flow_module()

    def flow_module(self) -> nn.Module:
        if type(self).flow_forward is not InvertibleMixin.flow_forward:
            module = copy.copy(self)
            module.forward = self.flow_forward
            return module
        raise TypeError(f"`flow_forward` is not defined for `{type(self)}`."
                        + " Either `flow_forward` or `flow_module` should be overloaded.")

    def flow_forward(self, *args, **kwargs):
        # overload either this or flow_module
        raise InverseError()

    def check_inverse(self: nn.Module, *inputs):
        for m in self.modules():
            register_inverse_check_hook(m)
        self(*inputs)


class LogJacobianMixin:  # TODO
    # Note: reconstruction_tols is used in inverse check.
    # It can be overridden with a property.
    reconstruction_tols = dict(rtol=1e-2, atol=1e-2)

    @functools.cached_property
    def inverse(self: nn.Module) -> nn.Module:
        # print(vmu.try_get_module_name_from_call_stack(self))
        try:
            inv_module = self.inverse_module()
            # The inverse of the inverse is the original module and `property` is used so that the
            # original module is not recognized as a child of the inverse module, which would result
            # in an infinite recursion when the `children` method is called on the inverse.
            inv_module.inverse = property(self._get_self)
        except AttributeError as e:
            # Turn it into a TypeError so that it doesn't get turned into a confusing
            # AttributeError saying that this module has no `inverse` attribute
            raise TypeError(f"The inverse for the module `{type(self)}` is not defined: {e}")
        return inv_module

    def _get_self(self):
        return self

    def inverse_module(self) -> nn.Module:
        if hasattr(self, 'inverse_forward'):
            return Inverse(self)
        raise TypeError(f"An inverse for the module `{type(self)}` is not defined.")

    def check_inverse(self: nn.Module, *inputs):
        for m in self.modules():
            register_inverse_check_hook(m)
        self(*inputs)

    def is_invertible(self) -> bool:
        return hasattr(self, 'inverse_forward') or hasattr(self, 'inverse_module')


class InitializableMixin:
    def initialize(self: nn.Module, *args, **kwargs):
        if len(*args) + len(**kwargs) != 0:
            y = self(*args, **kwargs)
        if hasattr(self, '_init') and not hasattr(type(self), '_init'):
            if len(vuf.params(self.init)) == 1:
                self.init(self)
            else:
                self.init(self, *args, **kwargs)
        return y


# Core Modules #####################################################################################

def is_built(module, including_submodules=False):
    return all(not hasattr(m, "is_built") or m.is_built(False)
               for m in itertools.chain([module], module.modules() if including_submodules else []))


@replaces('Module')
class Module(nn.Module, SplittableMixin, InvertibleMixin, ABC):
    # Based on https://github.com/MagNet-DL/magnet/blob/master/magnet/nodes/nodes.py
    def __init__(self):
        super().__init__()
        self._built = False
        self._check = None

    def is_built(self, including_submodules=False):
        if not self.is_built():
            return False
        if including_submodules:
            return all(not hasattr(m, "is_built") or m.is_built(False) for m in self.modules())

    def __call__(self, *args, **kwargs):
        # Modified to support in-place modification checking and shape inference
        try:
            if self._built:
                if hasattr(self, '_check'):  # checks are performed on the second input
                    check_hash = hash(tuple(_extract_tensors(*args, **kwargs)))
                    if self._check is None:
                        self._check = check_hash
                    elif self._check != check_hash:
                        # check on the second unique input, after post_build of the previous module
                        return self._call_with_check(*args, **kwargs)
                return super().__call__(*args, **kwargs)
            else:
                device = _try_get_device_from_args(*args, **kwargs)
                if type(self).build != Module.build:
                    self.build(*args, **kwargs)
                if device is not None:
                    self.to(device)
                if type(self).post_build != Module.post_build:
                    super().__call__(*args, **kwargs)
                    self.post_build(*args, **kwargs)
                    if device is not None:
                        self.to(device)
                self._built = True
                return super().__call__(*args, **kwargs)
        except Exception as e:
            print(
                f"Error in {vmu.try_get_module_name_from_call_stack(self)}, {type(self).__name__}")
            raise e

    # In-place modification of input checking

    def _mark_if_modified(self, out, inp_to_ver):
        if isinstance(out, torch.Tensor):
            return mark_modified(out, out._version != inp_to_ver.get(out, out._version))
        elif isinstance(out, T.Sequence):
            return type(out)(self._mark_if_modified(o, inp_to_ver) for o in out)
        elif isinstance(out, T.Mapping):
            return type(out)((k, self._mark_if_modified(o, inp_to_ver)) for k, o in out.items())

    def _call_with_check(self, *args, **kwargs):
        self._check_input(*args, **kwargs)  # single check after the parent is built
        del self._check
        inp_to_ver = {a: a._version for a in _extract_tensors(*args, **kwargs)}
        return self._mark_if_modified(super().__call__(*args, **kwargs), inp_to_ver)

    def _check_input(self, *args, **kwargs):
        """Checks whether the input is not in-place modified.

        This is evaluated when the module is called a second time,
        i.e. when the parent is built. It should not to modify the module.
        """
        if is_modified(args[0]):
            module_name = vmu.try_get_module_name_from_call_stack(self)
            raise RuntimeError(f"The input of {module_name} is in-place modified.")

    # Initialization with shape inference

    def store_args(self, attribute_name='args', args=None):
        """Can be called from the __init__ method after super().__init__ to
        store the arguments that the constructor/initializer was called with."""
        args = args if args is not None else class_initializer_locals_c()
        if attribute_name in [None, '']:
            self.__dict__.update(args)
        else:
            setattr(self, attribute_name, NameDict(args))

    def build(self, *args, **kwargs):
        """This is run before the first evaluation"""
        pass

    def post_build(self, *args, **kwargs):
        """This is run after the first evaluation (if overridden)."""
        pass

    # Modified standard nn.Module methods

    def add(self, *args, **kwargs):
        """A generalization of add_module that can accept multiple modules if
        supplied as keyword arguments."""
        if len(args) == 2 and len(kwargs) == 0:
            super().add_module(*args)
        elif len(args) == 0 and len(kwargs) > 0:
            for name, module in dict(*args, **kwargs).items():
                super().add_module(name, module)
        else:
            raise RuntimeError("Either only 2 positional arguments (name, module) or no positional"
                               + " and at least 1 keyword argument need to be supplied.")

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


@replaces('Identity')
class Identity(Module, nn.Identity):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def inverse_module(self):
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


@replaces('ModuleList')
class ModuleTable(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        args, kwargs = _to_sequential_init_args(*args, **kwargs), {}
        if len(args) == 1 and isinstance(args[0], T.Mapping):
            for key, module in args[0].items():
                self.add(key, module)
        else:
            for idx, module in enumerate(args):
                self.add(str(idx), module)

    def index(self, key):
        """Returns index of a child module from its name or the module itself."""
        elements = list(zip(*self._modules.items()))[int(not isinstance(key, str))]
        try:
            return elements.index(key)
        except ValueError as e:
            if isinstance(key, str):
                raise ValueError(f'The Seq contains no module named "{key}", only {elements}.\n{e}')

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(itertools.islice(iterator, idx, None))

    def _idx_to_canonical_form(self, idx):
        try:  # convert slice with str bound to slice with int bounds
            if isinstance(idx, slice) and (isinstance(idx.start, str) or isinstance(idx.stop, str)):
                children_names = list(zip(*self.named_children()))[0]
                return slice(*(children_names.index(i) if isinstance(i, str) else i
                               for i in (idx.start, idx.stop)), idx.step)
        except ValueError:
            raise KeyError(f"Invalid index: {idx}.")
        return idx

    @property
    def _slice_class(self):
        return Seq

    def __getitem__(self, idx):
        idx = self._idx_to_canonical_form(idx)
        if isinstance(idx, slice):
            return self._slice_class(dict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return self._modules[idx]
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx) if isinstance(idx, int) else idx
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self._modules.keys()[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())


@replaces('Sequential')
class Seq(ModuleTable, nn.Sequential):
    """A wrapper around torch.nn.Seq to enable passing a dict as the only
    parameter whereas in torch.nn.Seq only OrderedDict is accepted
    currently.
    It also supports slicing using strings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoints = None

    @property
    def _slice_class(self):
        return Seq

    def forward(self, x):
        modules = [x for x in self._modules.values()]
        cp_iter = iter(self._checkpoints or ())
        cp_range = next(cp_iter, None)
        i = 0
        while i < len(self):
            if cp_range is not None and cp_range[0] == i:
                def run_segment(x_, cp_range_=cp_range):
                    for j in range(cp_range_[0], cp_range_[1] + 1):
                        x_ = modules[j](x_)
                    return x_

                if torch.is_grad_enabled():
                    checkpoint = vtu.StateAwareCheckpoint(modules[cp_range[0]: cp_range[1] + 1])
                    x = checkpoint(run_segment, x)
                else:
                    x = run_segment(x)
                i, cp_range = cp_range[1] + 1, next(cp_iter, None)
            else:
                x = modules[i](x)
                i += 1
        return x

    def inverse_module(self):
        return Seq({k: m.inverse for k, m in reversed(self._modules.items())})

    def deep_split(self, submodule_path):
        next_name, path_remainder = submodule_path[0], submodule_path[1:]
        split_index = self.index(next_name)
        left, right = self[:split_index + 1], self[split_index + int(len(path_remainder) == 0):]
        if len(path_remainder) > 0:
            left[-1] = deep_split(left[-1], path_remainder)[0]
            right[0] = deep_split(right[0], path_remainder)[1]
        return left, right

    def deep_join(self, other):
        if type(other) is not Seq:
            raise ValueError("other must be of type Seq.")

        def index_to_name(module, index):
            return list(module.named_children())[index][0]

        if len(self) * len(other) == 0 or index_to_name(self, -1) != index_to_name(other, 0):
            return self.join(other)  # shallow joining suffices
        self_copy = self[:]
        self_copy[-1] = deep_join(self_copy[-1], other[0])
        return self_copy.join(other[1:])

    def split(self, submodule_name):
        ind = self.index(submodule_name)
        return self[:ind], self[ind:]

    def join(self, other):
        return Seq(dict(itertools.chain(self.named_children(), other.named_children())))

    def set_checkpoints(self, *inclusive_ranges):
        self._checkpoints = [(self.index(idx),) * 2 if isinstance(idx, str) else
                             tuple(map(self.index, idx)) if isinstance(idx[0], str) else
                             (idx,) * 2 if isinstance(idx[0], int) else
                             idx for idx in inclusive_ranges]
        max = -1
        for i, c in enumerate(self._checkpoints):
            if c[0] <= max or c[1] < c[0] or c[0] >= len(self):
                raise IndexError(f"Invalid sequence of checkpoint ranges: {self._checkpoints}."
                                 + f" Error at index {i} ({c}).")
            max = c[1]

    def clear_checkpoints(self):
        self._checkpoints = None


# Fork, parallel, reduction, ... ###################################################################


class Fork(ModuleTable):
    def forward(self, input):
        return tuple(m(input) for m in self)

    def inverse_module(self):
        return Fork({k: m.inverse() for k, m in self.named_children()})


class Parallel(ModuleTable):
    def forward(self, *inputs):
        inputs = vmu.sole_tuple_to_varargs(inputs)
        if len(self) == 1:
            return [self[0](x) for x in inputs]
        elif len(inputs) != len(self):
            raise ValueError(f"The number of inputs ({len(inputs)}) does not"
                             + " match the number of parallel modules."
                             + f"\nError in {vmu.try_get_module_name_from_call_stack(self)}.")
        return tuple(m(x) for m, x in zip(self, inputs))


class Merge(nn.Module):
    def forward(self, *inputs):
        inputs = vmu.sole_tuple_to_varargs(inputs)
        result = []
        for x in inputs:
            (result.extend if isinstance(x, tuple) else result.append)(x)
        return tuple(result)


class TupleSplit(Module):
    def __init__(self, split_indices):
        super().__init__()
        self.split_indices = split_indices

    def forward(self, x):
        result = []
        last_si = 0
        for si in self.split_indices:
            result.append(x[last_si:si])
            last_si = si
        result.append(x[self.split_indices[-1]:])
        return tuple(result)


class Reduce(Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    def forward(self, *inputs):
        inputs = vmu.sole_tuple_to_varargs(inputs)
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
        inputs = vmu.sole_tuple_to_varargs(inputs)
        shape = inputs[0].shape
        for oo in inputs[1:]:
            if oo.shape != shape:
                print(vmu.try_get_module_name_from_call_stack(self),
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
        self.dim, self.sizes = dim, None

    def build(self, *inputs):
        inputs = vmu.sole_tuple_to_varargs(inputs)
        self.sizes = [x.shape[self.dim] for x in inputs]

    def forward(self, *inputs):
        inputs = vmu.sole_tuple_to_varargs(inputs)
        return torch.cat(inputs, self.dim)

    def inverse_module(self):
        return Split(self.sizes, dim=self.dim)


class Split(Module):
    def __init__(self, split_size: T.Union[int, T.Sequence], dim=1):
        super().__init__()
        self.split_size_or_sections, self.dim = split_size, dim

    def forward(self, x: torch.Tensor):
        return x.split(self.split_size_or_sections, dim=self.dim)

    def inverse_module(self):
        return Concat(self.dim)


class Chunk(Module):
    def __init__(self, chunk_count: int, dim=1):
        super().__init__()
        self.store_args(store_in_self=True)

    def forward(self, x):
        return x.chunk(self.chunk_count, dim=self.dim)

    def inverse_module(self):
        return Concat(self.dim)


class Permute(Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def inverse_module(self):
        dims = self.dims
        inv_dims = [-1] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        return Permute(*inv_dims)


class Transpose(Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dims = (dim0, dim1)

    def forward(self, x):
        return x.transpose(*self.dims)

    def inverse_module(self):
        return self


class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, (x.shape[0], *self.shape))


class BatchReshape(Module):
    def __init__(self, *shape_or_func: T.Union[tuple, T.Callable[[tuple], tuple]]):
        super().__init__()
        if len(shape_or_func) == 1 and callable(shape_or_func[0]):
            shape_or_func = shape_or_func[0]
        self.shape_or_func = shape_or_func
        self.shape_inv = dict()

    def _get_out_shape(self, x):
        in_shape = x.shape[1:]
        out_shape = sof(*in_shape) if callable(sof := self.shape_or_func) else sof
        if (n_in := np.prod(in_shape)) != (n_out := np.prod(out_shape)):
            raise RuntimeError(
                f"The output shape ({out_shape}, {n_out} elements) requires a different number of"
                + f" elements than the input shape ({in_shape}, {n_in} elements).")
        self.shape_inv[out_shape] = in_shape
        return out_shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self._get_out_shape(x))

    def inverse_forward(self, y):
        return y.reshape(y.shape[0], *self.shape_inv[y.shape[1:]])


def _parse_auto_reshape_arg(dims_or_factors):
    dims_or_factors = re.findall(r" *(\([^)]*\)|[^(),]*) *(?:,|$)", dims_or_factors)
    return [-1 if x.strip() != '-1' else
            _parse_auto_reshape_arg(x[1:-1]) if x[0] == '(' else
            Fraction(x[1:].strip()) if x[0] == '*' else
            int(x) for x in dims_or_factors]


class AutoReshape(Module):
    """A reshape module that can be adaptive to input shape."""

    def __init__(self, dims_or_factors: T.Union[str, T.Sequence]):
        super().__init__()
        if isinstance(dims_or_factors, str):
            self.dims_or_factors = _parse_auto_reshape_arg(dims_or_factors)

    def build(self, x):
        def get_subshape(d, dims_or_factors):
            other = d // np.prod(f for f in dims_or_factors if f != -1)

            return sum([int(d * f) if isinstance(f, Fraction) else
                        int(d * (1 - other)) if f == -1 else
                        f for f in dims_or_factors], [])

        self.shape = [get_subshape(d, f) if isinstance(f, T.Sequence) else
                      [int(d * f) if isinstance(f, Fraction) else f] for d, f in
                      zip(x.shape, self.dims_or_factors)]

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(*self.shape)

    def inverse_module(self):
        return BatchReshape(*self.orig_shape)


class Rearrange(Module):  # TODO: einops.layers.torch import Rearrange
    def __init__(self, expr: str):
        super().__init__()
        self.expr = expr

    def forward(self, x):
        return einops.rearrange(self.expr, x)

    def inverse_module(self):
        return Rearrange("->".join(reversed(self.expr.split("->"))))


class Contiguous(Module):
    def forward(self, x):
        return x.contiguous()

    def inverse_module(self):
        return Identity()


class InvContiguous(Identity):
    def forward(self, x):
        return x

    def inverse_module(self):
        return Contiguous()


class Index(Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.__getitem__(*self.args)


class To(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, kwargs

    def forward(self, x):
        return x.to(*self.args, **self.kwargs)


class Clamp(nn.Module):
    def __init__(self, min_=None, max_=None, inplace=False):
        super().__init__()
        self.min_, self.max_, self.inplace = min_, max_, inplace

    def forward(self, x):
        x_ = mark_modified(x, self.inplace)
        return torch.clamp(x_, min=self.min_, max=self.max_, out=x_ if self.inplace else None)


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


# in-place modification marking

def mark_modified(x, mark=True):
    """Adds a `modified=True` attribute to the input tensor and returns a new
    tensor, a view on the same array without the attribute.

    Arguments:
        x (Tensor): input tensor.
        mark (bool): whether to set the modified attribute or just return the
            input. This optional argument is for convenience so that it can be
            used like `f(mark_modified(x, inplace), inplace=inplace))` instead
            of `f(mark_modified(x) if inplace else x, inplace=inplace)`.

    Example:
        >>> x = torch.randn(5,5)
        >>> x_ = mark_modified(x)
        >>> assert x_ is not x and torch.all(x_ == x)
        >>> assert is_modified(x) and not is_modified(x_)
        >>> y = x_.relu_()  # an in-place operation should be applied to x_
    """
    if mark:
        setattr(x, 'modified', True)
        return x[...]
    return x


def is_modified(x):
    return hasattr(x, 'modified')


# Wraps all modules and functions with inplace to support the "modified" annotation

def _forward_method_with_mark_modified(method):
    @functools.wraps(method)
    def forward(self, x, *args, **kwargs):
        return method(self, mark_modified(x, self.inplace), *args, **kwargs)

    return forward


def _func_with_mark_modified(func):
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        return func(mark_modified(x, kwargs['inplace']), *args, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


def _wrap_torch_operations(namespace):
    for name, v in vars(F).items():
        if not name.startswith('_') and callable(v) and 'inplace' in vuf.params(v):
            namespace[name] = _func_with_mark_modified(v)
            replaces(name)
    for name, v in vars(M).items():
        if not name.startswith('_') and inspect.isclass(v) and issubclass(v, nn.Module) \
                and 'inplace' in vuf.params(v):
            namespace[name] = type(name, (v,),
                                   {'forward': _forward_method_with_mark_modified(v.forward)})
            replaces(name)


_wrap_torch_operations(vars())


# Wrapped modules ##################################################################################

def _dimensional_build(name, input, args, in_channels_name='in_channels') -> nn.Module:
    if in_channels_name in args and args[in_channels_name] is None:
        args[in_channels_name] = input.shape[1]
    dim = len(input.shape) - 2  # assuming 1 batch and 1 channels dimension
    if dim not in [1, 2, 3]:
        raise ValueError(f"Cannot infer {name} dimension from input shape.")
    name = f"{name}{dim}d"
    layer_func = nn.__getattribute__(name)
    for k in vuf.params(layer_func).keys():
        if k not in args:
            raise ValueError(f"Missing argument for {name}: {k}.")
    module = layer_func(**args)
    return module


def _get_conv_padding(padding_type, kernel_size, dilation):
    if not isinstance(padding_type, str):
        return padding_type
    if any(k % 2 == 0 for k in ([kernel_size] if isinstance(kernel_size, int) else kernel_size)):
        raise ValueError(f"`kernel_size` must be an odd positive integer "
                         f"or a sequence of them, not {kernel_size}.")
    if padding_type not in ('half', 'full'):
        raise ValueError(f"Invalid padding_type value {padding_type}.")

    def get_padding(k, d):
        return (k - 1) * d // 2 if padding_type == 'half' else (k - 1) * d

    if any(isinstance(x, T.Sequence) for x in [kernel_size, dilation]):
        if isinstance(dilation, int):
            dilation = [dilation] * len(kernel_size)
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(dilation)
        return tuple(get_padding(k, d) for k, d in zip(kernel_size, dilation))
    else:
        return get_padding(kernel_size, dilation)


class WrappedModule(Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x):
        return self.orig(x)

    def __repr__(self):
        return "W" + repr(self.orig)


# TODO: Make separate Conv*ds
@replaces(*(f'Conv{i}d' for i in range(1, 4)))
class Conv(WrappedModule):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=None, in_channels=None, padding_mode='zeros'):
        if bias is None:
            raise ValueError("The bias argument should be provided for the Conv module.")
        padding = _get_conv_padding(padding, kernel_size, dilation)
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.orig = _dimensional_build("Conv", x, self.args)


class DeconvConv(WrappedModule):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=None, in_channels=None, padding_mode='zeros'):
        if bias is None:
            raise ValueError("The bias argument should be provided for the Conv module.")
        padding = _get_conv_padding(padding, kernel_size, dilation)
        if padding_mode != 'zeros':
            raise NotImplemented("padding_mode other than 'zeros' not supported.")
        del padding_mode
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        if self.args.in_channels is None:
            self.args.in_channels = x.shape[1]
        self.orig = FastDeconv(**self.args)


@replaces(*(f'MaxPool{i}d' for i in range(1, 4)))
class MaxPool(WrappedModule):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        padding = _get_conv_padding(padding, kernel_size, dilation)
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.orig = _dimensional_build("MaxPool", x, self.args)


@replaces(*(f'AvgPool{i}d' for i in range(1, 4)))
class AvgPool(WrappedModule):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        padding = _get_conv_padding(padding, kernel_size, dilation=1)
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.orig = _dimensional_build("AvgPool", x, self.args)


@replaces(*(f'ConvTranspose{i}d' for i in range(1, 4)))
class ConvTranspose(WrappedModule):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=1, groups=1,
                 bias=True, dilation=1, in_channels=None):
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.orig = _dimensional_build("ConvTranspose", x, self.args)


@replaces('Linear')
class Linear(WrappedModule):
    def __init__(self, out_features: int, bias=True, in_features=None):
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.args.in_features = self.args.in_features or np.prod(x.shape[1:])
        self.orig = nn.Linear(**{k: v for k, v in self.args.items()})

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.view(x.size(0), -1)
        return super().forward(x)


@replaces(*(f'BatchNorm{i}d' for i in range(1, 4)))
class BatchNorm(WrappedModule):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 num_features=None):
        super().__init__(orig=None)
        self.store_args()

    def build(self, x):
        self.orig = _dimensional_build("BatchNorm", x, self.args, 'num_features')


class GhostBatchNorm(BatchNorm):
    # Based on https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
    def __init__(self, batch_size, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_features=None):
        super().__init__(eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats, num_features=num_features)
        self.store_args()
        self.running_mean = self.running_var = self.num_splits = None

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


# Additional generally useful M ##############################################################


class _Func(Module):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class Func(_Func):
    def __init__(self, func, func_inv=None, module=None):
        super().__init__(func)
        self._inv = func_inv
        self.module = (module if module else func if isinstance(func, nn.Module) else None)

    def inverse_module(self):
        if self._inv is None:
            raise RuntimeError("Inverse not defined.")
        return Func(self._inv, self._func, module=self.module)


class Inverse(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.inverse_forward(*args, **kwargs)

    def flow_module(self):
        return self.module.flow.inverse

    def inverse_module(self):
        return self.module


class Flow(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.flow_forward(*args, **kwargs)

    def inverse_module(self):  # TODO
        return self.module.inverse.flow


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


@replaces('Dropout')
class Dropout(_DropoutNd):
    def forward(self, input):
        return F.dropout(mark_modified(input, self.inplace), self.p,
                         training=self.training or self.stochastic_eval, inplace=self.inplace)


@replaces('Dropout2d')
class Dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout2d(mark_modified(input, self.inplace), self.p,
                           training=self.training or self.stochastic_eval, inplace=self.inplace)


class AdditiveGaussianNoise(StochasticModule):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x).mul_(self.std) \
            if self.training or self.stochastic_eval else x


# Utilities ########################################################################################

class Adapter(Module):
    def __init__(self, module_or_factory, input_adapter=None, output_adapter=None):
        super().__init__()
        mof = module_or_factory
        self.module = mof if isinstance(mof, nn.Module) else mof()
        self.input_adapter = input_adapter or vuf.identity
        self.output_adapter = output_adapter or vuf.identity

    def forward(self, x):
        x = self.input_adapter(x) if self.input_adapter else x
        y = self.module(x)
        return self.output_adapter(y) if self.output_adapter else y


def parameter_count(module) -> Namespace:
    trainable, non_trainable = 0, 0
    for _, p in module.named_parameters():
        n = np.prod(p.size())
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return Namespace(trainable=trainable, non_trainable=non_trainable)


def get_submodule(root_module, path: T.Union[str, T.Sequence]) -> T.Union[Module, torch.Tensor]:
    """
    Returns a submodule of `root_module` that corresponds to `path`. It works
    for other attributes (e.g. Parameters) too.
    Arguments:
        root_module (Module): a module.
        path (Tensor): a string with the name of the module relative to
            `root_module`.
    """
    if isinstance(path, str):
        path = [] if path == '' else path.split('.')
    for name in path:
        if not hasattr(root_module, name):
            built_message = ("is not fully initialized (built) and"
                             if isinstance(root_module, Module) and not root_module.is_built() else "")
            raise AttributeError(
                f"The '{type(root_module).__name__}' instance {built_message}"
                + f" has no submodule '{name}'. It has"
                + f" children: {', '.join(list(k for k, v in root_module.named_children()))}.")
        root_module = getattr(root_module, name)
    return root_module


def deep_split(root: nn.Module, submodule_path: T.Union[list, str]):
    if isinstance(submodule_path, str):
        submodule_path = [] if submodule_path == '' else submodule_path.split('.')
    if len(submodule_path) == 0:
        return root, Seq()
    if not hasattr(root, 'deep_split'):
        raise NotImplementedError(f"Splitting not implemented for module type {type(root)}")
    return root.deep_split(submodule_path)


def deep_join(left: Module, right: Module):
    if not hasattr(left, 'deep_join'):
        raise NotImplementedError(f"Joining not implemented for module type {type(left)}")
    return left.deep_join(right)


def with_intermediate_outputs(module: nn.Module, submodule_paths: list = None,
                              inplace_modified_action: T.Literal['warn', 'error', None] = 'warn',
                              return_dict=False):
    """Creates a function wrapping `module` that returns a pair containing the
    output of `module.forward` as well as a list of intermediate outputs as
    defined by `submodule_paths`.

    Arguments:
        module (Module): a module.
        submodule_paths (List[str]): a list of names (relative to `root`)
            of modules the outputs of which you want to get.
        inplace_modified_action: What to do if it is detected that an
            intermediate output is in-place modified by a subsequent
            operation.

    Example:
        >>> module(x)
        tensor(...)
        >>> module_wio = with_intermediate_outputs(module, ['backbone', 'head.logits'])
        >>> module_wio(x)
        tensor(...), (tensor(...), tensor(...))
    """

    if submodule_paths is None:
        submodule_paths = [k for k, _ in module.named_modules()]
    elif isinstance(submodule_paths, str):
        submodule_paths = [submodule_paths]

    def get_submodules():
        return [get_submodule(module, p) for p in submodule_paths]

    @functools.wraps(module)
    def wrapper(*args, **kwargs):
        try:
            submodules = get_submodules()
        except AttributeError as ex:
            module(*args, **kwargs)  # in case the module is not yet built
            submodules = get_submodules()

        outputs = [None] * len(submodule_paths)

        def create_hook(idx):
            def hook(module, input, output):
                outputs[idx] = output

            return hook

        handles = [m.register_forward_hook(create_hook(i)) for i, m in enumerate(submodules)]
        output = module(*args, **kwargs)
        for h in handles:
            h.remove()

        if inplace_modified_action:
            for smp, o in zip(submodule_paths, outputs):
                if is_modified(o):
                    message = f"The (intermediate) output of {smp} is" \
                              + f" in-place modified by a subsequent operation."
                    if inplace_modified_action.startswith('warn'):
                        warnings.warn(message)
                    else:
                        raise RuntimeError(message)

        outputs = dict(zip(submodule_paths, outputs)) if return_dict else tuple(outputs)
        return output, outputs

    return wrapper


def with_intermediate_outputs_tree(
        module: nn.Module, submodule_paths=None,
        inplace_modified_action: T.Literal['warn', 'error', None] = 'warn', leaf_name='out'):
    """Creates a function wrapping `module` that returns a pair containing the
    output of `module.forward` as well as a tree of intermediate outputs as
    defined by `submodule_paths`.

    Arguments:
        module (Module): a module.
        submodule_paths (optional, List[str]): a list of names (relative to
            `root`) of modules the outputs of which you want to get. When the
             value is `None` (default), outputs of all submodules are stored.
        inplace_modified_action: What to do if it is detected that an
            intermediate output is in-place modified by a subsequent
            operation.

    Example:
        >>> module(x)
        tensor(...)
        >>> module_wiot = with_intermediate_outputs(module)
        >>> module_wiot(x)
        tensor(...), {'block1': {'conv': tensor(...), ...}, ...}
    """

    wio = with_intermediate_outputs(
        module, submodule_paths=submodule_paths,
        inplace_modified_action=inplace_modified_action, return_dict=True)

    @functools.wraps(module)
    def wrapper(*args, **kwargs):
        output, outputs = wio(*args, **kwargs)
        path_to_value = (((*k.split('.'), leaf_name), v) for k, v in outputs.items())
        return output, tree.unflatten(path_to_value)

    return wrapper


class IntermediateOutputsModuleWrapper(Module):
    def __init__(self, module, submodule_paths,
                 inplace_modified_action: T.Literal['warn', 'error', None] = 'warn'):
        """A wrapper module that returns a pair containing the output of
        `module.__call__` as well as a list of intermediate outputs as defined
        by `submodule_paths`.

        Arguments:
            module (Module): a module.
            submodule_paths (List[str]): a list of names (relative to `root`)
                of modules the outputs of which you want to get.
            inplace_modified_action: What to do if it is detected that an
                intermediate output is in-place modified by a subsequent
                operation.
        """
        super().__init__()
        self.module = module
        self.submodule_paths = submodule_paths

    def forward(self, *args, **kwargs):
        module_wio = with_intermediate_outputs(self.module, self.submodule_paths)
        return module_wio(*args, **kwargs)


class CheckpointingModuleWrapper(Module):
    def __init__(self, module, checkpoint=cp.checkpoint):
        super().__init__()
        self.module = module
        self.checkpoint = checkpoint

    def forward(self, *args):
        return self.checkpoint(self.module, *args)


# Gradient modification

class _RevGrad(torch.autograd.Function):
    """From https://github.com/janfreyberg/pytorch-revgrad"""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return -grad_output if ctx.needs_input_grad[0] else None


rev_grad = _RevGrad.apply


class RevGrad(Module):
    def forward(self, x):
        return rev_grad(x)


class _AmpGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, a):
        ctx.save_for_backward(a)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        a, = ctx.saved_variables
        return a * grad_output if ctx.needs_input_grad[0] else None


amp_grad = _AmpGrad.apply


class AmpGrad(Module):
    def forward(self, x):
        return amp_grad(x)


class StopGrad(Module):
    def forward(self, x):
        return x.detach()
