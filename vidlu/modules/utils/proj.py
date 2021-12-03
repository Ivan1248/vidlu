import dataclasses as dc
import typing as T
import torch

import vidlu.modules as vm
import vidlu.ops as vo


class ParameterProjector:
    def __call__(self, module, input=None):
        raise NotImplemented

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input, output):
            if not repeat:
                handle.remove()
            self(module, input)

        handle = module.register_forward_hook(hook)
        return handle


@dc.dataclass
class ClampProjector(ParameterProjector):
    param_path_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, module, input=None):
        for path, bounds in self.param_path_to_bounds.items():
            vo.clamp(vm.get_submodule(module, path), *bounds, inplace=True)


@dc.dataclass
class ScalingProjector(ParameterProjector):
    param_path_to_radius: T.Mapping[str, float]
    dim: T.Union[int, T.Sequence[int]] = -1
    p: int = 2

    def __call__(self, module, input=None):
        for path, radius in self.param_path_to_radius.items():
            par = vm.get_submodule(module, path)
            norm = par.norm(self.p, dim=self.dim, keepdim=True)
            par.mul_(torch.where(norm > radius, radius / norm, norm.new_ones(())))


@dc.dataclass
class CombinedProjector(ParameterProjector):
    projectors: T.List[ParameterProjector]

    def __call__(self, module, input=None):
        for proj in self.projectors:
            proj(module, input)

    def register_as_pre_forward_hook(self, module, repeat=False):
        def hook(module, input, output):
            if not repeat:
                handle.remove()
            for proj in self.projectors:
                proj(module, input)

        handle = module.register_forward_hook(hook)
        return handle
