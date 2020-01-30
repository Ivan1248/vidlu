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
        super().__init__(gamma=vmi.AlterLogGamma((2, 3)),
                         lightness=vmi.Multiplicative((2, 3)),
                         bias=vmi.Additive(()),
                         warp=vmi.Warp(),
                         soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01)))


class AlgTohchabTorbiwasc(vm.Seq, vmi.PerturbationModel):
    def __init__(self):
        super().__init__(gamma=vmi.AlterLogGamma((2, 3)),
                         to_hsv=vm.Func(voi.rgb_to_hsv),
                         cw_bias=vmi.Additive((2, 3)),
                         to_rgb=vm.Func(voi.hsv_to_rgb),
                         bias=vmi.Additive(()),
                         warp=vmi.Warp(),
                         soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01)))
