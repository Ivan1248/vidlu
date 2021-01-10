import dataclasses as dc
import typing as T
import torch

import vidlu.modules as vm
import vidlu.ops as vo


class Initializer:
    def __call__(self, pert_model, x):
        raise NotImplemented


@dc.dataclass
class UniformInit(Initializer):
    name_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, pert_model, x):
        for path, bounds in self.name_to_bounds.items():
            vo.random_uniform_(vm.get_submodule(pert_model, path), *bounds)


@dc.dataclass
class NormalInit(Initializer):
    name_to_mean_std: T.Mapping[str, T.Tuple[float, float]]

    def __call__(self, pert_model, x):
        for path, (mean, std) in self.name_to_mean_std.items():
            vm.get_submodule(pert_model, path).normal_(mean=mean, std=std)


class MultiInit(Initializer):
    def __init__(self, *args, **kwargs):
        self.initializers = dict(*args, **kwargs)

    def __call__(self, pert_model, x):
        if isinstance(self.initializers, T.Mapping):
            for name, init in self.initializers.items():
                init(vm.get_submodule(pert_model, name), x)
        else:
            for init in self.initializers:
                init(pert_model, x)

    def __getitem__(self, item):
        return self.initializers[item]

    def __getattr__(self, item):
        return self[item]
