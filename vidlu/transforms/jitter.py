from dataclasses import dataclass

from .image import RandomCrop, RandomHFlip, Pad
from vidlu.utils.func import compose


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

class CifarPadRandomCropHFlip(ClassificationJitter):
    def apply_input(self, x):
        return compose(Pad(4), RandomCrop(x[0].shape[-2:]), RandomHFlip())(x)


@dataclass
class SegRandomCropHFlip(SegmentationJitter):
    crop_shape: tuple

    def apply(self, x):
        return compose(RandomCrop(self.crop_shape), RandomHFlip())(tuple(x))


@dataclass
class SegRandomHFlip(SegmentationJitter):
    def apply(self, x):
        return RandomHFlip()(tuple(x))
