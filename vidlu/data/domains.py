import torch
import vidlu.utils.text as vut


# Domain types

class Domain:
    pass


class Name(str, Domain):
    pass


class Info(Domain):
    __slots__ = 'item'

    def __init__(self, obj):
        self.item = obj


class Array(torch.Tensor, Domain):
    def __new__(cls, obj):
        return obj[:].as_subclass(cls)

    def check_validity(self):
        pass


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
