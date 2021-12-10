import typing as T
import warnings
import collections

import torch
from torch.utils.data.dataloader import default_collate as torch_collate
import numpy as np

import vidlu.utils.text as vut

# Domain types

_type_to_domain = dict()


class Domain:
    @classmethod
    def collate(cls, elements, general_collate=None):
        if general_collate is None:
            raise NotImplementedError()
        else:
            return general_collate(elements)


class Name(str, Domain):
    @classmethod
    def collate(cls, elements, general_collate=None):
        return elements


class Other(Domain):
    __slots__ = 'item'

    def __init__(self, obj):
        self.item = obj

    @classmethod
    def collate(cls, elements, general_collate=None):
        return elements


class Array(torch.Tensor, Domain):
    def __new__(cls, obj):
        return obj[:].as_subclass(cls)

    def check_validity(self, quick):
        return True

    @classmethod
    def collate(cls, elements, general_collate=None):
        return torch_collate(elements)


class Spatial2D(Domain):
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


class AABB(Spatial2D):
    """Represents an axis-aligned bounding box.

    The min and max attributes store horizontal-vertical coordinate pairs (x, y).
    Bounds are considered to be on edges of the bounding box.

    The center of the first pixel of an image is considered to be at (0.5, 0.5). A bounding box that
    covers the first pixel of an image has min=(0, 0), max=(1, 1), and size=(1, 1) (size=(width, height).
    """
    __slots__ = 'min', 'max'

    def __init__(self, min, *, size=None, max=None):
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
        return f'{type(self).__name__}({tuple(self.min)}, {tuple(self.max)})'

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


class Float(Array, Domain):
    def check_validity(self, quick=False):
        return (super().check_validity(quick)
                and self.ndim == 0
                and torch.is_floating_point(self))


# Mapping from keys in snake case to Domain subclasses and vice versa
from_key = {vut.to_snake_case(name): cls for name, cls in globals().items()
            if isinstance(cls, type) and issubclass(cls, Domain)}
to_key = {cls: name for name, cls in from_key.items()}
