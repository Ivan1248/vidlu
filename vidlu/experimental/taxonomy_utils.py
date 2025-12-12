# Experimental. TODO
"""
A toxonomy is a collection of superclass-subclass declarations.
A superclass-subclass declaration is a tuple `(superclass, subclass)`,
where `superclass` is a class identifier (string) and `subclass` collection of
identifiers of classes that are disjoint subsets of `superclass`.
"""
import dataclasses as dc
import typing as T


# Class (set)

class ClassBase:
    pass


class StrClass(str, ClassBase):
    pass


@dc.dataclass(eq=True, order=True)
class Class(ClassBase):
    a: object


def to_class(c):
    return c if isinstance(c, ClassBase) else Class(c)


class_types = (str, ClassBase)
TClass = T.Union[str, ClassBase]


# Relations

@dc.dataclass(eq=True, order=True)
class BinarySetRelation:
    a: TClass
    b: TClass


@dc.dataclass
class Equality(BinarySetRelation):
    pass


@dc.dataclass
class Subset(BinarySetRelation):
    pass


@dc.dataclass
class PartialOverlap(BinarySetRelation):
    pass


# Operations

class ClassOperation(ClassBase):
    pass


@dc.dataclass
class BinaryClassOperation(ClassBase):
    a: TClass
    b: TClass


class NaryClassOperation(tuple, ClassBase):
    def __eq__(self, other):
        return type(other) == type(self) and super().__eq__(other)


@dc.dataclass
class Difference(BinaryClassOperation):
    pass


class Intersection(NaryClassOperation):
    pass


def intersection(a, b):
    a = to_class(a)
    b = to_class(b)
    if isinstance(a, Union) and isinstance(b, Union):
        disjoint_a = []
        disjoint_b = []


class Union(NaryClassOperation):
    pass


def split(a: TClass, b: TClass) -> T.Sequence[TClass]:
    return [Difference(a, b), Intersection(a, b), Difference(b, a)]
