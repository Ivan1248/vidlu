from dataclasses import dataclass, InitVar


@dataclass(init=True, repr=True)
class A:
    a: int


@dataclass
class B(A):
    a: int = 1  # default argument doesn't work if the type annotation is missing


@dataclass
class C(A):
    a = 1  # works


B()  # works
C()  # __init__ expects argument a


###

@dataclass(init=True, repr=True)
class A:
    a: int


@dataclass
class B(A):
    a = InitVar  # this should not obe allowed


###

from dataclasses import dataclass, InitVar


@dataclass
class A:
    attr: object = None  # no error if no default value


@dataclass
class B(A):
    pass


@dataclass
class C(B):  # Inherited non-default argument(s)defined in B follows inherited default argument defined in A
    pass