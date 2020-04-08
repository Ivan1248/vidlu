from dataclasses import dataclass
import warnings
import typing as T
import torch

from vidlu.transforms.image import RandomCrop, RandomHFlip, Pad, RandomScaleCrop, PadToShape
from vidlu.utils.func import compose
import vidlu.transforms.image as vti

from .randaugment import RandAugment as PILRandAugment


class SegmentationJitter:
    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        raise NotImplementedError()


class ClassificationJitter:
    def __call__(self, x):
        return self.apply_input(x[0]), self.apply_label(x[1])

    def apply_input(self, x):
        return x

    def apply_label(self, y):
        return y


####################################################################################################

class CifarPadRandCropHFlip(ClassificationJitter):
    def apply_input(self, x):
        return compose(Pad(4), RandomCrop(x[0].shape[-2:]), RandomHFlip())(x)


class SegRandHFlip(SegmentationJitter):
    def apply(self, x):
        return RandomHFlip()(tuple(x))


@dataclass
class SegRandCropHFlip(SegmentationJitter):
    crop_shape: tuple

    def apply(self, x):
        return compose(RandomCrop(self.crop_shape), RandomHFlip())(tuple(x))


@dataclass
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
        x, y = RandomHFlip()(xy)
        return (PadToShape(self.shape, value=self.image_pad_value)(x),
                PadToShape(self.shape, value=self.label_pad_value)(y))


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
