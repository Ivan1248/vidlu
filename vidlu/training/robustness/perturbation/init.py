import dataclasses as dc
import typing as T
import torch

import vidlu.modules as vm
import vidlu.ops as vo


class Initializer:
    def __call__(self, pert_model, x):
        raise NotImplemented


@dc.dataclass
class LInfBallUniformInitializer(Initializer):
    param_path_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, pert_model, x):
        for path, bounds in self.param_path_to_bounds.items():
            vo.random_uniform_(vm.get_submodule(pert_model, path), *bounds)


@dc.dataclass
class NormalInitializer(Initializer):
    param_path_to_mean_std: T.Mapping[str, T.Tuple[float, float]]

    def __call__(self, pert_model, x):
        for path, (mean, std) in self.param_path_to_mean_std.items():
            vm.get_submodule(pert_model, path).normal_(mean=mean, std=std)


@dc.dataclass
class CombinedInitilizer(Initializer):
    initializers: T.List[Initializer]

    def __call__(self, pert_model, x):
        for init in self.initializers:
            init(pert_model, x)
