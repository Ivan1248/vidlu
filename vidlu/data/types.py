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

    def check_validity(self):
        pass

    @classmethod
    def collate(cls, elements, general_collate=None):
        return torch_collate(elements)


class Spatial2D(Domain):
    pass


class ArraySpatial2D(Array, Spatial2D):
    def check_validity(self):
        super().check_validity()
        assert self.ndim in {3, 4}
        assert self.min() >= 0 and self.max() <= 1


class Image(ArraySpatial2D):
    def check_validity(self):
        super().check_validity()
        assert torch.is_floating_point(self)


class HSVImage(ArraySpatial2D):
    pass


class SegMap(ArraySpatial2D):
    def check_validity(self):
        super().check_validity()
        assert not torch.is_floating_point(self)


class AABB(Spatial2D):
    """Describes an axis-aligned bounding box.

    The min and max attributes store horizontal-vertical coordinate pairs (x, y).

    A bounding box that covers the first pixel has min=(0, 0), max=(1, 1), and size=(1, 1).
    The center of the first pixel is at (0.5, 0.5).

    Bounds are inclusive."""
    __slots__ = 'min', 'size'

    def __init__(self, min, *, size=None, max=None):
        warnings.warn("(x, y) vs (y, x), why is Torch HW and not WH")
        self.min = np.array(min)
        self.size = np.array(max) - self.min if size is None else np.array(size)

    @property
    def max(self):
        return self.min + self.size

    @property
    def center(self):
        return self.min + self.size / 2

    def __repr__(self):
        return f'{type(self).__name__}({tuple(self.min)}, {tuple(self.max)})'

    def __eq__(self, other):
        return np.all(self.min == other.min) and np.all(self.size == other.size)

    def __add__(self, translation):
        if not isinstance(translation, np.ndarray):
            translation = np.array(translation)
        return type(self)(min=self.min + translation, size=self.size)

    def __mul__(self, scale):
        return type(self)(min=self.min * scale, size=self.size * scale)

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

    def check_validity(self):
        for x in self:
            assert isinstance(x, AABB)
        assert len(self.shape) == 2

    def map(self, func, shape=None):
        return type(self)(map(func, self), shape=shape or self.shape)


class ClassAABBsOnImage(dict, AABBsOnImageCollection):
    """A list of AABBs with image size ((width, height)).
    """

    def __init__(self, *args: T.Mapping[object, T.Sequence[AABB]], shape: T.Sequence = None):
        super().__init__(*args)
        self.shape = None if shape is None else np.array(shape)

    def __repr__(self):
        return f"{type(self).__name__}({dict.__repr__(self)}, shape={self.shape!r})"

    def check_validity(self):
        for aabbs in self.values():
            for x in aabbs:
                assert isinstance(x, AABB)
        assert len(self.shape) == 2

    def map(self, func, shape=None):
        return type(self)({k: list(map(func, aabbs)) for k, aabbs in self.items()},
                          shape=shape or self.shape)


class ClassLabelLike(Array):
    def check_validity(self):
        super().check_validity()
        assert self.ndim in {0, 1}
        assert not torch.is_floating_point(self)


class ClassLabel(ClassLabelLike):
    def check_validity(self):
        super().check_validity()
        assert self.ndim == 0
        assert not torch.is_floating_point(self)


class ClassDist(ClassLabelLike):
    def check_validity(self):
        super().check_validity()
        assert self.ndim == 0
        assert torch.is_floating_point(self)
        assert torch.isclose(self.sum(), 1).item()


class Float(Array, Domain):
    def check_validity(self):
        super().check_validity()
        assert self.ndim == 0
        assert torch.is_floating_point(self)


# Mapping from keys in snake case to Domain subclasses and vice versa
from_key = {vut.to_snake_case(name): cls for name, cls in globals().items()
            if isinstance(cls, type) and issubclass(cls, Domain)}
to_key = {cls: name for name, cls in from_key.items()}
