from functools import partial

from torch import nn
import torch.nn.functional as F

import vidlu.modules as vm
import vidlu.modules.components as vmc
import vidlu.modules.inputwise as vmi
import vidlu.transforms.image as vti
import vidlu.ops as vo


class Galiebwasc(vmi.PerturbationModel, vm.Seq):
    def __init__(self):
        super().__init__(gamma=vmi.AlterLogGamma((1, 1)),
                         lightness=vmi.Multiplicative((1, 1)),
                         bias=vmi.Additive(()),
                         warp=vmi.Warp(),
                         soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01)))


class GaTohchabTorbiwasc(vmi.PerturbationModel, vm.Seq):
    def __init__(self):
        super().__init__(gamma=vmi.AlterLogGamma((1, 1)),
                         to_hsv=vm.Func(vti.rgb_to_hsv),
                         cw_bias=vmi.Additive((1, 1)),
                         to_rgb=vm.Func(vti.hsv_to_rgb),
                         bias=vmi.Additive(()),
                         warp=vmi.Warp(),
                         soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01)))
