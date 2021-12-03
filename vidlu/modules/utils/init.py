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
        def hook(module, input, output):
            if not repeat:
                handle.remove()
            self(module, input)

        handle = module.register_forward_hook(hook)
        return handle


@dc.dataclass
class UniformInit(Initializer):
    name_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, module, input=None):
        for path, bounds in self.name_to_bounds.items():
            vo.random_uniform_(vm.get_submodule(module, path), *bounds)


@dc.dataclass
class NormalInit(Initializer):
    name_to_mean_std: T.Mapping[str, T.Tuple[float, float]]

    def __call__(self, module, input=None):
        for path, (mean, std) in self.name_to_mean_std.items():
            vm.get_submodule(module, path).normal_(mean=mean, std=std)


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

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input, output):
            if not repeat:
                handle.remove()
            if isinstance(self.initializers, T.Mapping):
                for name, init in self.initializers.items():
                    init.register_as_pre_forward_hook(vm.get_submodule(module, name))
            else:
                for init in self.initializers:
                    init.register_as_pre_forward_hook(module)

        handle = module.register_forward_hook(hook)
        return handle
