from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from functools import reduce

import torch
from torch import nn
import torch.functional as F
import numpy as np

from vidlu.utils.collections import NameDict
from vidlu.utils.misc import clean_locals_from_first_initializer, find_frame_in_call_stack, get_key
from vidlu.utils.func import params, tryable


# Module class extenders ###########################################################################

def _scoped(*superclasses):
    class ScopedModuleMixin(*superclasses):
        # def __del__(self):
        #    for c in self.children():
        #        self._unregister_as_parent(c)

        def _register_as_parent(self, child_name, child_module):
            if hasattr(child_module, 'parents'):
                child_module.parents[self] = child_name
            else:
                child_module.parents = {self: child_name}

        def _unregister_as_parent(self, child_module):
            del child_module.parents[self]

        def __setattr__(self, key, value):
            super().__setattr__(key, value)
            # if isinstance(value, nn.Module):
            #    self._register_as_parent(key, value)

        def add_module(self, *args, **kwargs):
            if len(args) == 2 and len(kwargs) == 0:
                name, module = args
            elif len(args) == 0 and len(kwargs) == 1:
                name, module = next(kwargs.items())
            else:
                raise ValueError(
                    "Either a pair of positional arguments or single keyword argument can be accepted.")
            super().add_module(name, module)
            self._register_as_parent(name, module)

        @property
        def scopei(self):
            if hasattr(self, 'parents'):
                scopes = [f'{par.scope}.{name}' for par, name in self.parents.items()]
                if len(scopes) == 1:
                    return scopes[0]
                return '(' + ', '.join(scopes) + ')'
            # breakpoint()
            return 'ROOT'

        def get_parents(self):
            import gc
            for module in [r for r in gc.get_objects() if isinstance(r, (nn.Module))]:
                for k, v in module.named_children():
                    if v is self:
                        yield module, k
            """for r_modules in [r for r in gc.get_referrers(self)
                              if isinstance(r, (dict, OrderedDict))]:
                found = False
                for r__dict__ in [r for r in gc.get_referrers(r_modules)
                                  if type(r) is dict and r.get('_modules', None) is r_modules]:
                    for r_parent in gc.get_referrers(r__dict__):
                        if isinstance(r_parent, nn.Module) and r__dict__ is r_parent.__dict__:
                            yield r_parent, get_key(r_modules, self)
                            found = True
                            break
                    if found:
                        break"""

        @property
        def scope(self):
            scopes = [f'{par.scope}.{name}' for par, name in self.get_parents()]
            if len(scopes) == 1:
                return scopes[0]
            return '(' + ', '.join(scopes) + ')'

    return ScopedModuleMixin


def _stochastic(*superclasses):
    class StochasticModuleMixin(*superclasses):
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

    return StochasticModuleMixin


def _extended(*superclasses):
    return superclasses
    #return [_stochastic(_scoped(*superclasses))]


# Modified PyTorch Modules #########################################################################

class Module(*_extended(nn.Module, ABC)):
    # Based on https://github.com/MagNet-DL/magnet/blob/master/magnet/nodes/nodes.py

    def __init__(self):
        super().__init__()
        args = clean_locals_from_first_initializer()
        for x in ['self', '__class__']:
            args.pop(x, None)
        self.args = NameDict(args)
        self._built = False

    def __call__(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        result = super().__call__(*args, **kwargs)
        if not self._built:
            recompute_result = self.post_build(*args, **kwargs) is None
            self._built = True
            if recompute_result:
                return super().__call__(*args, **kwargs)
        return result

    def __str__(self):
        return (self.__class__.__name__ + "("
                + ", ".join(f"{k}={v}" for k, v in self.args.items()) + ")")

    def clone(self):
        return self.__class__(**self.args)

    @property
    def device(self):
        param = next(self.parameters(), None)
        return None if param is None else param[0].device

    def load_state_dict(self, state_dict_or_path, strict=True):
        """ Handle a path being given instead of a file. (preferred since it
        automatically maps to the correct device). Taken from MagNet. """
        from pathlib import Path
        sd = state_dict_or_path
        if isinstance(sd, (str, Path)):
            sd = torch.load(sd, map_location=self.device)
        return super().load_state_dict(sd, strict=strict)

    def build(self, *args, **kwargs):
        pass

    def post_build(self, *args, **kwargs):
        """
        Leave this as it is if you don't need more initialization after
        evaluaton. I you do, override it and make it return nothing (`None`).
        """
        return True


def _to_sequential_init_args(*args, **kwargs):
    if len(kwargs) > 0:
        if len(args) > 0:
            raise ValueError(
                "If keyword arguments are supplied, no positional arguments are allowed.")
        args = [kwargs]
    if len(args) == 1 and isinstance(args[0], dict):
        args = [OrderedDict(args[0])]
    return args


class Identity(Module):
    def forward(self, x):
        return x


class Parallel(*_extended(nn.ModuleList)):
    def forward(self, *input):
        return [m(*input) for m in self]


class Reduce(Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    def forward(self, input):
        return reduce(self.func, input[1:], input[0])


class Sequential(*_extended(nn.Sequential)):
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
                children_names = list(zip(*self.named_children()))
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
        return list(zip(*self.named_children())).index(key)


class EModuleDict(nn.ModuleDict):
    def __init__(self, modules: dict, **kwargs):
        modules.update(kwargs)
        super().__init__(modules)


def _dimensional_build(name, input, args, in_channels_name='in_channels') -> nn.Module:
    if in_channels_name in args and args[in_channels_name] is None:
        args[in_channels_name] = input.shape[1]
    layer_func = nn.__getattribute__(f"{name}{len(input.shape) - 2}d")
    for k, v in params(layer_func).items():
        if k not in args:
            raise ValueError(f"Missing argument: {k}.")
    return layer_func(**args)


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


class Conv(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, in_channels=None):
        padding = (_get_conv_padding(padding, kernel_size, dilation)
                   if isinstance(padding, str) else padding)
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("Conv", x, self.args)

    def forward(self, x):
        return self.orig(x)


class MaxPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        super().__init__()
        self._padding = (_get_conv_padding(padding, kernel_size, dilation)
                         if isinstance(padding, str) else padding)
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("MaxPool", x, self.args)

    def forward(self, x):
        return self.orig(x)


class AvgPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super().__init__()
        self._padding = (_get_conv_padding(padding, kernel_size, dilation=1)
                         if isinstance(padding, str) else padding)
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("AvgPool", x, self.args)

    def forward(self, x):
        return self.orig(x)


class ConvTranspose(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=1, groups=1,
                 bias=True, dilation=1, in_channels=None):
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("ConvTranspose", x, self.args)

    def forward(self, x):
        return self.orig(x)


class Linear(Module):
    def __init__(self, out_features: int, bias=True, in_features=None):
        super().__init__()
        self.orig = None

    def build(self, x):
        self.args.in_features = self.args.in_features or np.prod(x.shape[1:])
        self.orig = nn.Linear(**{k: v for k, v in self.args.items()})

    def forward(self, x):
        if len(x.shape) != 2:
            x = x.view(x.size(0), -1)
        return self.orig(x)


class BatchNorm(Module):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 num_features=None):
        super().__init__()
        self.orig = None

    def build(self, x):
        self.orig = _dimensional_build("BatchNorm", x, self.args, 'num_features')

    def forward(self, x):
        return self.orig(x)


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
        list(list(zip(*inspect.getmembers(sys.modules[__name__])))[0]) + ['Module'] +
        [f'Conv{t}{d}d' for t in ['', 'Transpose'] for d in {1, 2, 3}])

for name, cls in inspect.getmembers(nn, inspect.isclass):
    if issubclass(cls, nn.Module) and name not in _already_wrapped_classes:
        class_definition = _class_template.format(typename=name)
        print(class_definition)
        exec(class_definition)
'''


# Additional generally useful modules ##############################################################

class Func(Module):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class Sum(EModuleDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input):
        return sum(m(*input) for m in self.items())


# Utils ############################################################################################

def parameter_count(module):
    from numpy import prod

    trainable, non_trainable = 0, 0
    for p in module.parameters():
        n = prod(p.size())
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return trainable, non_trainable


def get_submodule(root_module, path: str):
    """
    Returns a submodule of `root_module` that corresponds to `path`. It works
    for other attributes (e.g. Parameters) too.
    Arguments:
        root_module (Module): a module.
        path (Tensor): a string with the name of the module module relative to
        `root_module`
    """
    for name in [tryable(int, default_value=n)(n) for n in path.split('.')]:
        if isinstance(name, str):
            root_module = getattr(root_module, name)
        elif isinstance(name, int):
            root_module = root_module[name]
    return root_module


def split_on_module(root_module, split_path: str, include_left=False):
    next_name, *path_remainder = split_path.split('.', maxsplit=1)
    if len(path_remainder) > 0:
        path_remainder = path_remainder[0]
    else:
        if isinstance(root_module, Sequential):
            # TODO
            raise NotImplementedError()
            return Sequential


def with_intermediate_outputs(root_module: nn.Module, submodule_paths: list):
    """
    Creates a function extending `root_module.forward` so that a pair containing
    th output of `root_module.forward` as well as well as a list of intermediate
    outputs as defined in `submodule_paths`.
    Arguments:
        root_module (Module): a module.
        submodule_paths (List[str]): a list of names (relative to `root_module`)
        of modules the outputs of which you want to get.
    Example:
        >>> module(x)
        tensor(...)
        >>> module_wio = with_intermediate_outputs(module, 'backbone', 'head.logits')
        >>> module_wio(x)
        tensor(...), {'backbone': tensor(...), 'head.logits': tensor(...)}
    """
    submodules = [get_submodule(root_module, p) for p in submodule_paths]

    def forward(*args):
        outputs = [None] * len(submodule_paths)

        def create_hook(idx):
            def hook(module, input, output):
                outputs[idx] = output

            return hook

        handles = [m.register_forward_hook(create_hook(i)) for i, m in enumerate(submodules)]
        output = root_module(*args)
        for h in handles:
            h.remove()
        return output, dict(zip(submodule_paths, outputs))

    return forward


class IntermediateOutputsModuleWrapper(Module):
    def __init__(self, module, submodule_paths):
        """
        Creates a function extending `root_module.forward` so that a pair containing
        th output of `root_module.forward` as well as well as a list of intermediate
        outputs as defined in `submodule_paths`.
        Arguments:
            root_module (Module): a module.
            submodule_paths (List[str]): a list of module names relative to
            `root_module`.
        """
        super().__init__()
        self.module = module
        self.handles, self.outputs = None, None

    def __del__(self):
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

    def forward(self, *input):
        self.outputs = [None] * len(self.args.submodule_paths)
        output = self.module.forward(*input)
        return output, list(self.outputs)


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


# Debugging ########################################################################################

def get_calling_module(module, start_frame=None):
    def predicate(frame):
        locals_ = frame.f_locals
        if 'self' in locals_:
            self_ = locals_['self']
            return isinstance(locals_['self'], nn.Module) and self_ is not module

    frame = find_frame_in_call_stack(predicate, start_frame)
    caller = frame.f_locals['self']
    return caller, frame


def try_get_name_from_call_stack(module, full_name=True):
    parent, frame = get_calling_module(module)
    if parent is None:
        return ''
    for n, c in parent.named_children():
        if c is module:
            parent_name = try_get_name_from_call_stack(parent, frame.f_back)
            return f'{parent_name}.{n}' if full_name and parent_name else n
    breakpoint()
    raise NotImplementedError()
