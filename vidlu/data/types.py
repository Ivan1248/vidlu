import typing as T

import torch
from torch.utils.data.dataloader import default_collate as torch_collate
import numpy as np

import vidlu.utils.text as vut

from collections import UserList


# Modality types


class DataModality:
    @classmethod
    def collate(cls, elements, general_collate=None):
        if general_collate is None:
            raise NotImplementedError()
        else:
            return general_collate(elements)


class Array(torch.Tensor, DataModality):
    def __new__(cls, obj):
        return obj[...].as_subclass(cls)

    def check_validity(self, quick):
        return True

    @classmethod
    def collate(cls, elements, general_collate=None):
        shapes = [tuple(x.shape) for x in elements]
        if not all(s == shapes[0] for s in shapes[1:]):
            raise RuntimeError(f"All elements (type {type(elements[0]).__name__}) should have"
                               + f" equal shapes, but the shapes are {shapes}.")
        return torch_collate(elements)

    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     try:
    #         return super().__torch_function__(func, types, args, kwargs=kwargs)
    #     except TypeError as e:
    #         args = [v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
    #                 for v in args]
    #         kwargs = {k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
    #                   for k, v in kwargs.items()}
    #         return super().__torch_function__(func, types, args, kwargs=kwargs)


class ClassLabelLike(Array):
    def check_validity(self, quick=False):
        return (super().check_validity(quick)
                and self.ndim in {0, 1}
                and not torch.is_floating_point(self))


class ClassLabel(ClassLabelLike):
    def check_validity(self, quick=False):
        return (super().check_validity(quick)
                and self.ndim == 0
                and not torch.is_floating_point(self))


class ClassDist(ClassLabelLike):
    def check_validity(self, quick=False):
        return (super().check_validity(quick)
                and self.ndim == 0
                and torch.is_floating_point(self)
                and (quick or torch.isclose(self.sum(), 1).item()))


class Float(Array, DataModality):
    def check_validity(self, quick=False):
        return (super().check_validity(quick)
                and self.ndim == 0
                and torch.is_floating_point(self))


class Name(str, DataModality):
    @classmethod
    def collate(cls, elements, general_collate=None):
        return elements


class Other(DataModality):
    __slots__ = 'item'

    def __init__(self, obj):
        self.item = obj

    @classmethod
    def collate(cls, elements, general_collate=None):
        return elements


class Spatial2D(DataModality):
    pass


class ArraySpatial2D(Array, Spatial2D):
    def check_validity(self, quick=False):
        valid = super().check_validity(quick) and self.ndim in {3, 4}
        if quick:
            return valid
        return valid and self.min() >= 0 and self.max() <= 1


class Image(ArraySpatial2D):
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     result = super().__torch_function__(func, types, args=args, kwargs=kwargs)
    #     try:
    #         if result.shape[-3] > 4:
    #             result = result.as_subclass(torch.Tensor)
    #     except Exception:
    #         pass
    #     return result

    def check_validity(self, quick=False):
        return super().check_validity() and torch.is_floating_point(self)


class HSVImage(ArraySpatial2D):
    pass


class SegMap(ArraySpatial2D):
    def check_validity(self, quick=False):
        return super().check_validity() and not torch.is_floating_point(self)


class Mask2D(ArraySpatial2D):
    def check_validity(self, quick=False):
        return super().check_validity() and not torch.is_floating_point(self)


class SoftMask2D(ArraySpatial2D):
    def check_validity(self, quick=False):
        return super().check_validity() and torch.is_floating_point(self)


class ClassMasks2D(Spatial2D):
    __slots__ = 'classes', 'masks'

    def __init__(self, classes: T.Sequence[ClassLabel],
                 masks: T.Union[T.Sequence[Mask2D], T.Sequence[SoftMask2D]]):
        self.classes, self.masks = classes, masks


class AABB(Spatial2D):
    """Represents an axis-aligned bounding box.

    Bounds are considered to be on edges of the bounding box.

    The center of the first pixel of an image is considered to be at (0.5, 0.5). A bounding box that
    covers the first pixel of an image has min=(0, 0), max=(1, 1), and size=(1, 1)
    (size=(height, width).

    Recommended convention. The min and max attributes store row-column/height-width coordinate
    pairs. This corresponds to the shapes of arrays representing images and matrices in PyTorch,
    OpenCV and ImageIO, but does not correspond to the `size` attribute of PIL `Image`s and OpenCV
    `Point`s.
    """
    __slots__ = 'min', 'max'

    def __init__(self, min, *, max=None, size=None):
        self.min = np.array(min)
        self.max = self.min + np.array(size) if max is None else np.array(max)

    @property
    def size(self):
        return self.max - self.min

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def area(self):
        return np.prod(self.size)

    def __repr__(self):
        return f'{type(self).__name__}(min={tuple(self.min)}, max={tuple(self.max)})'

    def __eq__(self, other):
        return np.all(self.min == other.min) and np.all(self.max == other.max)

    def __add__(self, translation):
        if not isinstance(translation, np.ndarray):
            translation = np.array(translation)
        return type(self)(min=self.min + translation, max=self.max + translation)

    def __sub__(self, other):
        return self + (-other)

    # def __rsub__(self, other):  # error
    #     return type(self)(min=other - self.min, max=other - self.max)

    def __mul__(self, scale):
        return type(self)(min=self.min * scale, max=self.max * scale)

    __rmul__ = __mul__

    def map(self, func):
        return AABB(min=func(self.min), max=func(self.max))

    def clip(self, min=None, max=None):
        return type(self)(min=self.min.clip(min, max), max=self.max.clip(min, max))

    def transpose(self):
        return type(self)(min=list(reversed(self.min)), max=list(reversed(self.max)))

    def intersect(self, other, return_if_invalid=False):
        result = type(self)(min=np.maximum(self.min, other.min),
                            max=np.minimum(self.max, other.max))
        if return_if_invalid:
            return result
        return result if result.size.min() >= 0 else None

    @classmethod
    def collate(cls, elements):
        return elements


class AABBsOnImageCollection(Spatial2D):
    def map(self, func, shape=None):
        raise NotImplementedError()

    @classmethod
    def collate(cls, elements, general_collate=None):
        return elements


class AABBsOnImage(list, AABBsOnImageCollection):
    """A list of AABBs with image size ((width, height)).
    """

    def __init__(self, *args, shape: T.Sequence = None):
        super().__init__(*args)
        self.shape = None if shape is None else np.array(shape)

    def __repr__(self):
        return f"{type(self).__name__}({list.__repr__(self)}, shape={self.shape!r})"

    def check_validity(self, quick=False):
        return len(self.shape) == 2 and all(isinstance(x, AABB) for x in self)

    def map(self, func, shape=None):
        if shape is None:
            shape = self.shape
        return type(self)(map(func, self), shape=shape)


class ClassAABBsOnImage(dict, AABBsOnImageCollection):
    """A list of AABBs with image size ((width, height)).
    """

    def __init__(self, *args: T.Mapping[object, T.Sequence[AABB]], shape: T.Sequence = None):
        super().__init__(*args)
        self.shape = None if shape is None else np.array(shape)

    def __repr__(self):
        return f"{type(self).__name__}({dict.__repr__(self)}, shape={self.shape!r})"

    def check_validity(self, quick=False):
        return (len(self.shape) == 2
                and all(isinstance(x, AABB) for aabbs in self.values() for x in aabbs))

    def map(self, func, shape=None):
        if shape is None:
            shape = self.shape
        return type(self)({k: list(map(func, aabbs)) for k, aabbs in self.items()}, shape=shape)


# Based on Mask2Former code: Copyright (c) Facebook, Inc. and its affiliates.
class PaddedImageBatch(DataModality):
    """A tensor of images of possibly varying sizes (padded) together with the sizes."""

    def __init__(self, images, sizes):
        self.images = images
        self.sizes = sizes

    def get_mask(self, pos_value=1.0, neg_value=0.0):
        mask = torch.full_like(self.images, neg_value)
        for i, size in enumerate(self.sizes):
            mask[i, ..., :size[0], :size[1]] = pos_value

    def to_list(self):
        return [self[i, :, :size[0], :size[1]] for i, size in enumerate(self.sizes)]

    @staticmethod
    def from_list(images: T.List[torch.Tensor], size_divisibility: int = 0,
                  fill_value: float = 0.0) -> 'PaddedImageBatch':
        assert isinstance(images, (tuple, list))

        image_sizes = torch.tensor([list(x.shape[-2:]) for x in images])
        max_size = image_sizes.max(0).values
        min_size = image_sizes.max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        if tuple(max_size) == tuple(min_size):
            batched_images = torch.stack(images)
        else:
            batch_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
            batched_images = images[0].new_full(batch_shape, fill_value, device=images[0].device)
            for i, img in enumerate(images):
                batched_images[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)

        return PaddedImageBatch(batched_images, image_sizes)


# Mapping from keys in snake case to Domain subclasses and vice versa
from_key = {vut.to_snake_case(name): cls for name, cls in globals().items()
            if isinstance(cls, type) and issubclass(cls, DataModality)}
to_key = {cls: name for name, cls in from_key.items()}
