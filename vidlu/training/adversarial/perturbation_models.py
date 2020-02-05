from functools import partial

from torch import nn
import torch.nn.functional as F

import vidlu.modules as vm
import vidlu.modules.components as vmc
import vidlu.modules.inputwise as vmi
import vidlu.transforms.image as vti
import vidlu.ops.image as voi
import vidlu.ops as vo


class Alglibiwasc(vm.Seq, vmi.PerturbationModel):
    def __init__(self):
        super().__init__(partial(
            vm.Seq,
            gamma=vmi.AlterLogGamma((2, 3)),
            lightness=vmi.Multiplicative((2, 3)),
            bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01))))


class AlgTohchabTorbiwasc(vmi.PerturbationModelWrapper):
    def __init__(self):
        super().__init__(partial(
            vm.Seq,
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=vmi.PerturbationModelWrapper(vm.Func(voi.rgb_to_hsv), forward_arg_count=1),
            cw_bias=vmi.Additive((2, 3)),
            to_rgb=vmi.PerturbationModelWrapper(vm.Func(voi.hsv_to_rgb), forward_arg_count=1),
            # bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=vmi.PerturbationModelWrapper(
                vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01)), forward_arg_count=1)))
