from functools import partial
import dataclasses as dc
import typing as T

import torch
from torch import nn
import torch.nn.functional as F

import vidlu.modules as vm
import vidlu.modules.components as vmc
import vidlu.modules.inputwise as vmi
from vidlu.modules.inputwise import PerturbationModelWrapper as _PMW
import vidlu.transforms.image as vti
import vidlu.ops.image as voi
import vidlu.ops as vo


# Perturbation models

class Alglibiwasc(vm.Seq, vmi.PerturbationModel):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            lightness=vmi.Multiplicative((2, 3)),
            bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01))))


class AlgTohchabTorbiwasc(_PMW):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=_PMW(voi.rgb_to_hsv, forward_arg_count=1),
            cw_bias=vmi.Additive((2, 3)),
            to_rgb=_PMW(voi.hsv_to_rgb, forward_arg_count=1),
            # bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=_PMW(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01), forward_arg_count=1)))


class ChannelGammaHsv(_PMW):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=_PMW(voi.rgb_to_hsv, forward_arg_count=1),
            additive=vmi.Additive((2, 3)),
            to_rgb=_PMW(voi.hsv_to_rgb, forward_arg_count=1),
            soft_clamp=_PMW(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01), forward_arg_count=1)))


class ChannelGammaHsv(_PMW):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=_PMW(voi.rgb_to_hsv, forward_arg_count=1),
            additive=vmi.Additive((2, 3)),
            to_rgb=_PMW(voi.hsv_to_rgb, forward_arg_count=1),
            soft_clamp=_PMW(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01), forward_arg_count=1)))


# Initializers and projections


@dc.dataclass
class LInfBallUniformInitializer:
    param_path_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, pert_model):
        for path, bounds in self.param_path_to_bounds.items():
            vo.random_uniform_(vm.get_submodule(pert_model, path), *bounds)


@dc.dataclass
class NormalInitializer:
    param_path_to_mean_std: T.Mapping[str, T.Sequence[T.Tuple[float, float]]]

    def __call__(self, pert_model):
        for path, (mean, std) in self.param_path_to_mean_std.items():
            vm.get_submodule(pert_model, path).normal_(mean=mean, std=std)


@dc.dataclass
class ClampProjection:
    param_path_to_bounds: T.Mapping[str, T.Sequence[T.Union[float, torch.Tensor]]]

    def __call__(self, pert_model):
        for path, bounds in self.param_path_to_bounds.items():
            vo.clamp(vm.get_submodule(pert_model, path), *bounds, inplace=True)


@dc.dataclass
class ScalingProjection:
    param_path_to_radius: T.Mapping[str, float]
    dim: T.Union[int, T.Sequence[int]] = -1
    p: int = 2

    def __call__(self, pert_model):
        for path, radius in self.param_path_to_radius.items():
            par = vm.get_submodule(pert_model, path)
            norm = par.norm(self.p, dim=self.dim, keepdim=True)
            par.mul_(torch.where(norm > radius, radius / norm, norm.new_ones(())))
