from functools import partial
from numpy import s_

import vidlu.modules.inputwise as vmi
import vidlu.modules as vm
from vidlu.modules.inputwise import PertModel
import vidlu.ops.image as voi
import vidlu.ops as vo


# Perturbation models

class Alglibiwasc(vm.Seq, vmi.PertModelBase):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            lightness=vmi.Multiplicative((2, 3)),
            bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01))))


class AlgTohchabTorbiwasc(PertModel):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=PertModel(voi.rgb_to_hsv, forward_arg_count=1),
            ch_bias=vmi.Additive((2, 3)),
            to_rgb=PertModel(voi.hsv_to_rgb, forward_arg_count=1),
            # bias=vmi.Additive(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                 forward_arg_count=1)))


class ChannelGammaHsv(PertModel):
    def __init__(self, forward_arg_count=None):
        super().__init__(
            vm.Seq(
                gamma=vmi.AlterLogGamma((2, 3)),
                # mul=vmi.Multiplicative((2,3)),
                to_hsv=PertModel(voi.rgb_to_hsv, forward_arg_count=1),
                additive=vmi.Additive((2, 3), slice=(...,)),
                to_rgb=PertModel(voi.hsv_to_rgb, forward_arg_count=1),
                soft_clamp=PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                     forward_arg_count=1),
                # tps=vmi.BackwardTPSWarp()
            ),
            forward_arg_count=forward_arg_count)


class Photometric20(PertModel):
    def __init__(self, clamp=True, forward_arg_count=None):
        module = vm.Seq(to_hsv=PertModel(voi.rgb_to_hsv, forward_arg_count=1),
                        add_v=vmi.Additive((2, 3), slice=s_[:, 2:, ...]),
                        mul_s=vmi.Multiplicative((2, 3), slice=s_[:, 1:2, ...]),
                        add_h=vmi.Additive((2, 3), slice=s_[:, 0:1, ...]),
                        mul_v=vmi.Multiplicative((2, 3), slice=s_[:, 2:, ...]),
                        to_rgb=PertModel(voi.hsv_to_rgb, forward_arg_count=1),
                        shuffle=PertModel(partial(vm.shuffle, dim=1), forward_arg_count=1))
        if clamp:
            module.add(soft_clamp=PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                            forward_arg_count=1))
        super().__init__(module, forward_arg_count=forward_arg_count)


class PhotoTPS20(vm.Seq):
    def __init__(self, clamp=True, forward_arg_count=None):
        super().__init__(photometric=Photometric20(clamp, forward_arg_count),
                         tps=vmi.BackwardTPSWarp())
