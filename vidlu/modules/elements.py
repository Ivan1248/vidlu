from abc import ABC, abstractmethod
from argparse import Namespace
from collections import OrderedDict
import functools
from functools import reduce, partial
from typing import Union
from collections import Mapping, Sequence
import itertools
from os import PathLike

import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from vidlu.modules.utils import try_get_module_name_from_call_stack
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
                raise ValueError(
                    "Either a pair of positional arguments or single keyword argument can be accepted.")
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

    return ExtendedInterfaceModExt


def _extract_tensors(*args, **kwargs):
    tensors = []
    for a in itertools.chain(args, kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, (Mapping, Sequence)):
            for x in _extract_tensors(a):
                yield x


def _buildable(*superclasses):
    class BuildableModExt(*superclasses):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            args = class_initializer_locals_c()
            self.args = NameDict(args)
            self._built = False

        def __call__(self, *args, **kwargs):
            try:
                input_tensors = _extract_tensors(*args, **kwargs)
                device = tryable(lambda: list(input_tensors)[0].device, None)()
                if not self._built:
                    if type(self).build != BuildableModExt.build:
                        self.build(*args, **kwargs)
                        if device is not None:
                            self.to(device)
                    result = super().__call__(*args, **kwargs)
                    if type(self).post_build != BuildableModExt.post_build:
                        self.post_build(*args, **kwargs)
                        if device is not None:
                            self.to(device)
                        result = super().__call__(*args, **kwargs)
                    self._built = True
                else:
                    result = super().__call__(*args, **kwargs)
                return result
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

        def clone(self):
            return type(self)(**self.args)

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

        def _call_with_modifiers(self, name, *input):
            return self._modifiers.get(name, identity)(getattr(self, name))(*input)

    return ModifibleModExt


def _stochastic(*superclasses):
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
                pass
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


def _debuggable(*superclasses):
    class DebbugableModExt(*superclasses):

        def forward(self, *args, **kwargs):
            y = super().forward(*args, **kwargs)
            # print(try_get_module_name_from_call_stack(self), parameter_count(self))
            return y

    return DebbugableModExt


def _extended(*superclasses):
    return [e(*superclasses) for e in [_buildable, _extended_interface]]


class InvertibleModuleMixin:
    def inverse(self):
        raise NotImplementedError()


# Core Modules #####################################################################################

class Module(*_extended(nn.Module, ABC)):
    # Based on https://github.com/MagNet-DL/magnet/blob/master/magnet/nodes/nodes.py

    def __init__(self):
        super().__init__()
        if type(self).__init__ is not Module.__init__:
            args = class_initializer_locals_c()
            self.args = NameDict(args)
        self._built = False

    def __str__(self):
        return (self.__class__.__name__ + "("
                + ", ".join(f"{k}={v}" for k, v in self.args.items()) + ")")


class RevIdentity(nn.Identity, InvertibleModuleMixin):
    def reverse(self, y):
        return y

    def inverse(self):
        return RevIdentity()


def _to_sequential_init_args(*args, **kwargs):
    if len(kwargs) > 0:
        if len(args) > 0:
            raise ValueError(
                "If keyword arguments are supplied, no positional arguments are allowed.")
        args = [kwargs]
    if len(args) == 1 and isinstance(args[0], dict):
        args = [OrderedDict(args[0])]
    return args


# TODO: rename to Seq
class Sequential(*_extended(nn.Sequential), _modifiable(nn.Sequential)):
    """
    A wrapper around torch.nn.Sequential to enable passing a dict as the only
    parameter whereas in torch.nn.Sequential only OrderedDict is accepted
    currently.
    It also supports slicing using strings.
    """

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
            return Sequential(dict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return getattr(self, idx)
        return super().__getitem__(idx)

    def index(self, key):
        """Returns index of a child module from its name or the module itself."""
        return list(zip(*self.named_children()))[int(not isinstance(key, str))].index(key)

    def forward(self, *input):
        if len(self._modules) == 0 and len(input) != 1:
            raise RuntimeError("A `Sequential` with no children can only accept 1 argument.")
        for name in self._modules:
            input = (self._call_with_modifiers(name, *input),)
        return input[0]

    def inverse(self):
        result = Sequential({k: m.inverse() for k, m in reversed(self._modules.items())})
        result._modifiers = self._modifiers
        return result


class ModuleTable(Sequential):
    def forward(self, *input):
        raise NotImplementedError


# Fork, parallel and reduction ################################################################

def tuple_to_varargs_method(f):
    @functools.wraps(f)
    def wrapper(self, *inputs):
        if len(inputs) == 1:
            if isinstance(inputs[0], Sequence):  # tuple, not any sequence
                inputs = inputs[0]
        return f(self, *inputs)

    return wrapper


class Fork(ModuleTable):
    def forward(self, input):
        return tuple(m(input) for m in self)

    def inverhtose(self):
        return Fork({k.m.inverse() for k, m in self.named_children()})


class Parallel(ModuleTable):
    @tuple_to_varargs_method
    def forward(self, *inputs):
        if len(self) == 1:
            return [self[0](x) for x in inputs]
        elif len(inputs) != len(self):
            raise ValueError(f"The number of inputs ({len(inputs)}) does not"
                             + " match the number of parallel modules."
                             + f"\nError in {try_get_module_name_from_call_stack(self)}.")
        return tuple(m(x) for m, x in zip(self, inputs))


class Reduce(Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    @tuple_to_varargs_method
    def forward(self, *inputs):
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


class Sum(Module):
    def __init__(self):
        super().__init__()

    @tuple_to_varargs_method
    def forward(self, *inputs):
        shape = inputs[0].shape
        for oo in inputs[1:]:
            if oo.shape != shape:
                print(try_get_module_name_from_call_stack(self),
                      ' '.join(str(tuple(x.shape)) for x in inputs))
        y = inputs[0].clone()
        for x in inputs[1:]:
            y += x
        return y


class Concat(Module):
    def __init__(self, dim=1):
        super().__init__()

    @tuple_to_varargs_method
    def forward(self, *inputs):
        return torch.cat(inputs, self.args.dim)


# Wrapped modules ##################################################################################

def _dimensional_build(name, input, args, in_channels_name='in_channels') -> nn.Module:
    if in_channels_name in args and args[in_channels_name] is None:
        args[in_channels_name] = input.shape[1]
    dim = len(input.shape) - 2  # assuming 1 batch and 1 channels dimension
    if dim not in [1, 2, 3]:
        raise ValueError(f"Cannot infer {name} dimension from input shape.")
    name = f"{name}{dim}d"
    layer_func = nn.__getattribute__(name)
    for k, v in params(layer_func).items():
        if k not in args:
            breakpoint()
            raise ValueError(f"Missing argument for {name}: {k}.")
    module = layer_func(**args)
    return module


def _get_conv_padding(padding_type, kernel_size, dilation):
    assert all(k % 2 == 1
               for k in ([kernel_size] if isinstance(kernel_size, int) else kernel_size))
    if padding_type not in ('half', 'full'):
        raise ValueError(f"Invalid padding_type value {padding_type}.")

    def get_padding(k, d):
        k_ = 1 + (k - 1) * d
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
                 num_features=None):
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("BatchNorm", x, self.args, 'num_features')


class GhostBatchNorm(BatchNorm):
    # Based on https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
    def __init__(self, batch_size, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 num_features=None):
        super().__init__(eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats, num_features=num_features)
        self.batch_size = batch_size

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

    def forward(self, input):
        if self.training or not self.track_running_stats:
            N, C, *S = input.shape
            return F.batch_norm(
                input.view(-1, C * self.num_splits, *S), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(input.shape)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
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


class Func(Module):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class RevFunc(Func):
    def __init__(self, func, func_inv=None):
        super().__init__(func)
        self._func_inv = func_inv or None

    def reverse(self, y):
        if not self._func_inv:
            raise RuntimeError("The inverse function is not defined.")
        return self._func_inv(y)


# Stochastic #######################################################################################

class StochasticModule(Module, ABC):
    @abstractmethod
    def deterministic_forward(self, *args, **kwargs):
        pass


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
        return F.dropout(input, self.p, training=True, inplace=self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input):
        return F.dropout2d(input, self.p, training=True, inplace=self.inplace)


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
    from numpy import prod

    trainable, non_trainable = 0, 0
    for name, p in module.named_parameters():
        n = prod(p.size())
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
    for name in [tryable(int, default_value=n)(n) for n in path]:
        if isinstance(name, str):
            if not hasattr(root_module, name):
                raise AttributeError(
                    f"'{type(root_module).__name__}' has no submodule '{name}'. It has children:"
                    + f" {', '.join(list(k for k, v in root_module.named_children()))}.")
            root_module = getattr(root_module, name)
        elif isinstance(name, int):
            root_module = root_module[name]
    return root_module


def join_sequentials(a, b):
    return Sequential(**{k: v for k, v in a.named_children()},
                      **{k: v for k, v in b.named_children()})


def deep_split(root: nn.Module, split_path: Union[list, str]):
    if isinstance(split_path, str):
        split_path = [] if split_path == '' else split_path.split('.')
    if len(split_path) == 0:
        return root, Sequential()
    next_name, path_remainder = split_path[0], split_path[1:]
    if isinstance(root, Sequential):
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
    if not isinstance(left, Sequential):
        raise NotImplementedError(f"Joining not implemented for module type {type(left)}")

    def index_to_name(module, index):
        return list(module.named_children())[index][0]

    if len(left) * len(right) == 0 or index_to_name(left, -1) != index_to_name(right, 0):
        return join_sequentials(left, right)
    left = left[:]
    left[-1] = deep_join(left[-1], right[0])
    return join_sequentials(left, right[1:])


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

    def get_submodules(): return [get_submodule(root, p) for p in submodule_paths]

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
