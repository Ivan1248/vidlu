import dataclasses as dc
import typing as T
import torch

import vidlu.modules as vm
import vidlu.ops as vo


class Projector:
    def __call__(self, pert_model, x):
        raise NotImplemented


@dc.dataclass
class ClampProjector(Projector):
    param_path_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, module, x):
        for path, bounds in self.param_path_to_bounds.items():
            vo.clamp(vm.get_submodule(module, path), *bounds, inplace=True)


@dc.dataclass
class ScalingProjector(Projector):
    param_path_to_radius: T.Mapping[str, float]
    dim: T.Union[int, T.Sequence[int]] = -1
    p: int = 2

    def __call__(self, module, x):
        for path, radius in self.param_path_to_radius.items():
            par = vm.get_submodule(module, path)
            norm = par.norm(self.p, dim=self.dim, keepdim=True)
            par.mul_(torch.where(norm > radius, radius / norm, norm.new_ones(())))


@dc.dataclass
class CombinedProjector(Projector):
    projectors: T.List[Projector]

    def __call__(self, module, x):
        for proj in self.projectors:
            proj(module, x)
