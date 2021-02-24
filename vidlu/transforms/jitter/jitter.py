import dataclasses as dc
import warnings
import typing as T
import torch
from vidlu.utils.func import partial

from vidlu.transforms.image import RandomCrop, RandomHFlip, Pad, RandomScaleCrop, PadToShape
from vidlu.utils.func import compose
import vidlu.transforms.image as vti
import vidlu.modules as vm

from .randaugment import RandAugment as PILRandAugment


class Composition:
    def __init__(self, *jitters):
        self.jitters = jitters

    def __call__(self, x):
        for f in self.jitters:
            x = f(x)
        return x


class PhTPS20:
    def __init__(self):
        from vidlu.training.robustness import attacks
        from vidlu.training.robustness import perturbation as pert
        self.pert_model = pert.PhotoTPS20()
        self.init = pert.MultiInit(
            dict(tps=pert.NormalInit({'offsets': (0, 0.1)}),
                 photometric=pert.UniformInit(
                     {'module.add_v.addend': [-0.25, 0.25],
                      'module.mul_s.factor': [0.25, 2.],
                      'module.add_h.addend': [-0.1, 0.1],
                      'module.mul_v.factor': [0.25, 2.]})))

    def __call__(self, x):
        x, y = x
        x_ = x.unsqueeze(0)
        with torch.no_grad():
            if not vm.is_built(self.pert_model):
                self.pert_model(x_)
                for p in self.pert_model.parameters():
                    p.requires_grad = False
            self.init(self.pert_model, x_)
        x = self.pert_model(x_).squeeze(0)
        return x, y


class SegmentationJitter:
    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        raise NotImplementedError()


class ClassificationJitter:
    def __call__(self, x):
        if len(x) == 1:
            return (self.apply_input(x[0]),)
        return self.apply_input(x[0]), self.apply_label(x[1])

    def apply_input(self, x):
        return x

    def apply_label(self, y):
        return y


@dc.dataclass()
class ReplaceWithNoise(ClassificationJitter):
    mean: float = 0.
    std: float = 1.

    def apply_input(self, x):
        return torch.randn_like(x) * self.std + self.mean


####################################################################################################

class CifarPadRandCropHFlip(ClassificationJitter):
    def apply_input(self, x):
        return compose(Pad(4), RandomCrop(x[0].shape[-2:]), RandomHFlip())(x)


class SegRandHFlip(SegmentationJitter):
    def apply(self, x):
        return RandomHFlip()(tuple(x))


@dc.dataclass
class SegRandCropHFlip(SegmentationJitter):
    crop_shape: tuple

    def apply(self, x):
        return compose(RandomCrop(self.crop_shape), RandomHFlip())(tuple(x))


@dc.dataclass
class SegRandScaleCropPadHFlip(SegmentationJitter):
    shape: tuple
    max_scale: float
    overflow: object
    min_scale: float = None
    align_corners: bool = True
    image_pad_value: T.Union[torch.Tensor, float, T.Literal['mean']] = 'mean'
    label_pad_value = -1

    def apply(self, xy):
        xy = RandomScaleCrop(shape=self.shape, max_scale=self.max_scale,
                             min_scale=self.min_scale, overflow=self.overflow,
                             is_segmentation=(False, True),
                             align_corners=self.align_corners)(tuple(xy))
        xy = RandomHFlip()(xy)
        x = PadToShape(self.shape, value=self.image_pad_value)(xy[0])
        return (x,) if len(xy) == 1 else \
            (x, PadToShape(self.shape, value=self.label_pad_value)(xy[1]))


class RandAugment(ClassificationJitter):
    def __init__(self, n, m):
        self.pil_randaugment = PILRandAugment(n, m)

    def apply_input(self, x):
        with torch.no_grad():
            warnings.warn("RandAugment is not differentiable.")
            h = x.mul(255).to(torch.uint8)
            h = vti.torch_to_pil(h.permute(1, 2, 0).cpu())
            h = self.pil_randaugment(h)
            return vti.pil_to_torch(h).to(dtype=x.dtype, device=x.device).permute(2, 0, 1) / 256
