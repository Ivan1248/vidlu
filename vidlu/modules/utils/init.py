import dataclasses as dc
import typing as T
import torch

import vidlu.modules as vm
import vidlu.ops as vo


class Initializer:
    def __call__(self, module, input=None):
        raise NotImplemented

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = dict(state)

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input):
            if not repeat:
                handle.remove()
            self(module, input)

        handle = module.register_forward_pre_hook(hook)
        return handle


def get_param(module, path):
    param_or_module = vm.get_submodule(module, path)
    if not isinstance(param_or_module, torch.Tensor):
        params = dict(param_or_module.named_parameters())
        if (l := len(params)) > 1:
            raise ValueError(f"{path=} corresponds to a module which contains {l} > 1 parameters."
                             + f" This makes the parameter path ambiguous. The available parameters"
                             + f" are: " + ", ".join(params.keys()) + ".")
        elif l == 0:
            raise RuntimeError(f"The module {path} contains no parameters.")
        return next(iter(params.values()))
    return param_or_module


@dc.dataclass
class UniformInit(Initializer):
    name_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, module, input=None):
        for path, bounds in self.name_to_bounds.items():
            vo.random_uniform_(get_param(module, path), *bounds)


@dc.dataclass
class BinaryInit(Initializer):
    name_to_prob: T.Mapping[str, float]

    def __call__(self, module, input=None):
        for path, prob in self.name_to_prob.items():
            param = get_param(module, path)
            param.set_(torch.rand(()) <= prob)


@dc.dataclass
class NormalInit(Initializer):
    name_to_mean_std: T.Mapping[str, T.Tuple[float, float]]

    def __call__(self, module, input=None):
        for path, (mean, std) in self.name_to_mean_std.items():
            get_param(module, path).normal_(mean=mean, std=std)


class JointInit(Initializer):
    def __init__(self, *args):
        self.initializers = args

    def __call__(self, module, input=None):
        for init in self.initializers:
            init(module, input=input)

    def __getitem__(self, item):
        return self.initializers[item]

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input):
            if not repeat:
                handle.remove()
            for init in self.initializers:
                init.register_as_pre_forward_hook(module)

        handle = module.register_forward_pre_hook(hook)
        return handle

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(str, self.initializers))})"


class MultiInit(Initializer):
    def __init__(self, *args, **kwargs):
        self.initializers = dict(*args, **kwargs)

    def __call__(self, module, input=None):
        if input is None:
            if isinstance(self.initializers, T.Mapping):
                for name, init in self.initializers.items():
                    init(vm.get_submodule(module, name))
            else:
                for init in self.initializers:
                    init(module)
        else:
            self.register_as_pre_forward_hook(module)
            return module(input)

    def __getitem__(self, item):
        return self.initializers[item]

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        return f"{type(self).__name__}({self.initializers})"

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input):
            if not repeat:
                handle.remove()
            if isinstance(self.initializers, T.Mapping):
                for name, init in self.initializers.items():
                    init.register_as_pre_forward_hook(vm.get_submodule(module, name))
            else:
                for init in self.initializers:
                    init.register_as_pre_forward_hook(module)

        handle = module.register_forward_pre_hook(hook)
        return handle
