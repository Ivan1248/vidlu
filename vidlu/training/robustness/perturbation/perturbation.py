from vidlu.utils.func import partial
from numpy import s_

import vidlu.modules.inputwise as vmi
import vidlu.modules as vm
from vidlu.modules.inputwise import PertModel, SeqPertModel
import vidlu.ops.image as voi
import vidlu.ops as vo
import vidlu.utils.func as vmf
from vidlu.utils.collections import NameDict

t = vmf.ArgTree


# Perturbation models

class Alglibiwasc(vm.Seq, vmi.PertModelBase):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            lightness=vmi.Multiply((2, 3)),
            bias=vmi.Add(()),
            morsic_warp=vmi.MorsicTPSWarp(grid_shape=(2, 2), label_padding_mode='zeros'),
            soft_clamp=vm.Func(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01))))


class AlgTohchabTorbiwasc(PertModel):
    def __init__(self):
        super().__init__(vm.Seq(
            gamma=vmi.AlterLogGamma((2, 3)),
            to_hsv=PertModel(voi.rgb_to_hsv, forward_arg_count=1),
            ch_bias=vmi.Add((2, 3)),
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
                additive=vmi.Add((2, 3), slice=(...,)),
                to_rgb=PertModel(voi.hsv_to_rgb, forward_arg_count=1),
                soft_clamp=PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                     forward_arg_count=1),
                # tps=vmi.BackwardTPSWarp()
            ),
            forward_arg_count=forward_arg_count)


class Photometric20(SeqPertModel):
    def __init__(self, clamp, forward_arg_count=None):
        add_f = partial(vmi.Add, equivariant_dims=(2, 3))
        mul_f = partial(vmi.Multiply, equivariant_dims=(2, 3))
        modules = NameDict(to_hsv=PertModel(voi.rgb_to_hsv),
                           add_v=add_f(slice=s_[:, 2:, ...]),
                           mul_s=mul_f(slice=s_[:, 1:2, ...]),
                           add_h=add_f(slice=s_[:, 0:1, ...]),
                           mul_v=mul_f(slice=s_[:, 2:, ...]),
                           to_rgb=PertModel(voi.hsv_to_rgb),
                           shuffle=PertModel(partial(vm.shuffle, dim=1)))
        if clamp:
            modules.soft_clamp = PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                           forward_arg_count=1)
        super().__init__(**modules, forward_arg_count=forward_arg_count)


class PhotoTPS20(SeqPertModel):
    def __init__(self, clamp, forward_arg_count=None):
        super().__init__(photometric=Photometric20(clamp, forward_arg_count=1),
                         tps=vmi.BackwardTPSWarp(label_interpolation_mode='nearest',
                                                 label_padding_mode=-1),
                         forward_arg_count=forward_arg_count)


class PhotoWarp1(SeqPertModel):
    def __init__(self, clamp, forward_arg_count=None, sigma=3):
        super().__init__(warp=vmf.argtree_partial(vmi.SmoothWarp, smooth_f=t(sigma=sigma))(),
                         photometric=Photometric20(clamp, forward_arg_count=1),
                         forward_arg_count=forward_arg_count)


class Photometric3(SeqPertModel):
    def __init__(self, clamp, forward_arg_count=None):
        add_f = partial(vmi.Add, equivariant_dims=(2, 3))
        mul_f = partial(vmi.Multiply, equivariant_dims=(2, 3))
        modules = NameDict(to_hsv=PertModel(voi.rgb_to_hsv),
                           add_h=add_f(slice=s_[:, 0:1, ...]),
                           add_s=add_f(slice=s_[:, 1:2, ...]),
                           add_v=add_f(slice=s_[:, 2:, ...]),
                           mul_v=mul_f(slice=s_[:, 2:, ...]),
                           to_rgb=PertModel(voi.hsv_to_rgb))
        if clamp:
            modules.soft_clamp = PertModel(partial(vo.soft_clamp, min_=0, max_=1, eps=0.01),
                                           forward_arg_count=1)
        super().__init__(**modules, forward_arg_count=forward_arg_count)


class PhotoTPS3(SeqPertModel):
    def __init__(self, clamp, forward_arg_count=None):
        super().__init__(photometric=Photometric3(clamp, forward_arg_count=1),
                         tps=vmi.BackwardTPSWarp(label_interpolation_mode='nearest',
                                                 label_padding_mode=-1),
                         forward_arg_count=forward_arg_count)
