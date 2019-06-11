import inspect
from collections import Mapping
from inspect import signature
from functools import partialmethod, partial, reduce, wraps
import itertools

from .collections import NameDict
from vidlu.utils import tree


def identity(x):
    return x


# Wrappers #########################################################################################

def compose(func0, *funcs):
    """Creates a composition of functions.

    Functions are applied from left to right. The first function can have any
    signature, while other functions should be able to accept only 1 argument.

    Args:
        func0 (callable): the first function. I can have any signature.
        *funcs (callable): other functions. They are to be called with a single
            argument.
    """

    @wraps(func0)
    def wrapper(*args, **kwargs):
        x = func0(*args, **kwargs)
        for f in funcs:
            x = f(x)
        return x

    return wrapper


def pipe(x, *funcs):
    for f in funcs:
        x = f(x)
    return x


def do(proc, x):
    proc(x)
    return x


class hard_partial(partial):
    """
    Like partial, but doesn't allow changing already chosen keyword arguments.
    Strangely, works as desired, even though `partial.__new__` looks like it
    should copy the `keywords` attribute in case partial(hard_partial(f, x=2), x=3
    """

    def __call__(self, *args, **kwargs):
        for k in kwargs.keys():
            if k in self.keywords:
                raise RuntimeError(f"Parameter {k} is frozen and can't be overridden.")
        return partial.__call__(self, *args, **kwargs)


def freeze_nonempty_args(func):
    return hard_partial(func, **{k: v for k, v in default_args(func) if v is not Empty})


def tryable(func, default_value):
    def try_(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return default_value

    return try_


def _dummy(*a, **k):
    pass


class _FuncTree(partial, Mapping):
    def __new__(*args, **kwargs):
        cls, func, *args = args
        if func in [None, Empty]:
            p = partial.__new__(cls, _dummy, *args, **kwargs)
            p.__setattr__(('func', func))
            p.func = func
        return partial.__new__(cls, func, *args, **kwargs)

    def __getattr__(self, key):
        return self.keywords[key]

    def __eq__(self, other):
        if not isinstance(other, _FuncTree):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.keywords if isinstance(key, str) else 0 <= key < len(self.args)

    def __getitem__(self, key):
        return self.keywords[key] if isinstance(key, str) else self.args[key]

    def __iter__(self):
        return iter(self.keywords)

    def __len__(self):
        return len(self.keywords)

    def keys(self):
        return self.keywords.keys()

    def values(self):
        return self.keywords.values()

    def items(self):
        return self.keywords.items()


def functree(func, *args, **kwargs):
    kwargs = {k: functree(v) if callable(v) else v
              for k, v in itertools.chain(default_args(func).items(), kwargs.items())}
    return _FuncTree(func, **kwargs)


def functree_shallow(func, *args, **kwargs):
    return _FuncTree(func, **{**default_args(func), **kwargs})


# parameters/arguments #############################################################################

def parameter_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return len(signature(func).parameters)


def default_args(func) -> NameDict:
    return NameDict({k: v.default for k, v in signature(func).parameters.items()
                     if v.default is not Empty})


def params(func) -> NameDict:
    return NameDict({k: v.default for k, v in signature(func).parameters.items()})


def params_deep(func):
    for k, v in params(func).items():
        if callable(v):
            for k_, v_ in params_deep(v):
                yield [k] + k_, v_
        else:
            yield [k], v


def find_params_deep(func, predicate):
    for k, v in params(func).items():
        if callable(v):
            for k_, v_ in find_params_deep(v, predicate):
                yield [k] + k_, v_
        elif predicate(k, v):
            yield [k], v


def find_default_arg(func, arg_name_or_path, recursive=True, return_tree=False):
    path = [arg_name_or_path] if isinstance(arg_name_or_path, str) else arg_name_or_path
    found = dict()
    da = default_args(func)
    if path[0] in da:
        arg = da[path[0]]
        if len(path) == 1:
            found[path[0]] = arg
        elif callable(arg):
            result = find_default_arg(arg, path[1:], recursive=False)
            if len(result) > 0:
                found[path[0]] = result
        else:
            assert False
    if recursive:
        for k, v in da:
            if callable(v):
                result = find_default_arg(v, path)
                if len(result) > 0:
                    found[k] = result
    found = ArgTree(found)
    return found.items() if return_tree else tree.flatten(found)


def dot_default_args(func):
    for k, v in default_args(func).items():
        assert not hasattr(func, k)
        setattr(func, k, v)
    return func


def inherit_missing_args(parent_function):
    parent_default_args = default_args(parent_function)

    def decorator(func):
        inherited_args = {k: parent_default_args[k] for k, v in params(func).items()
                          if v is Empty and k in parent_default_args}
        return partial(func, **inherited_args)

    return decorator


# ArgTree, argtree_partial #########################################################################


class ArgTree(NameDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *other, **kwargs):
        if len(other) > 1:
            raise TypeError(f"update expected at most 1 positional argument, got {len(other)}.")
        other = other[0] if len(other) > 0 else {}
        for k, v in {**other, **kwargs}.items():
            if k not in self:
                self[k] = v
            if isinstance(self[k], ArgTree) and isinstance(v, ArgTree):
                self[k].update(v)
            elif callable(self[k]) and isinstance(v, ArgTree):
                self[k] = argtree_partial(self[k], v)
            else:
                self[k] = v

    """def deepen(self):
        for k in self:
            if callable(self[k]):
                self[k] = params_deep(self[k])
    """

    def copy(self):
        return ArgTree({k: v.copy() if isinstance(v, ArgTree) else v for k, v in self.items()})

    @staticmethod
    def from_func(func):
        return ArgTree(
            {k: ArgTree.from_func(v) if callable(v) and v not in (Empty, Reserved) else v
             for k, v in params(func).items()})


class HardArgTree(ArgTree):
    """An argtree whose arguments cannot be overridden."""


class EscapedArgTree:
    __slots__ = ('argtree',)

    def __init__(self, argtree):
        self.argtree = argtree


def argtree_partial(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument"
                         + " or 1 or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, HardArgTree):
            tree[k] = argtree_hard_partial(default_args(func)[k], v)
        if isinstance(v, ArgTree):
            tree[k] = argtree_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    if isinstance(tree, HardArgTree):
        return hard_partial(func, **tree)
    return partial(func, **tree)


def argtree_hard_partial(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) > 1:
        raise ValueError("The arguments should be either a single positional argument"
                         + " or 0 or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, ArgTree):
            tree[k] = argtree_hard_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    return hard_partial(func, **tree)


def argtree_partialmethod(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument "
                         + " or 1 or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, ArgTree):
            tree[k] = argtree_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    return partialmethod(func, **tree)


def find_params_tree(func, predicate):
    tree = dict()
    for k, v in params(func).items():
        if callable(v):
            subtree = find_params_tree(v, predicate)
            if len(subtree) > 0:
                tree[k] = subtree
        elif predicate(k, v):
            tree[k] = v
    return ArgTree(tree)


def find_empty_params_tree(func):
    return find_params_tree(func, predicate=lambda k, v: v is Empty)


def find_empty_params_deep(func):
    return find_params_deep(func, predicate=lambda k, v: v is Empty)


# Dict map, filter #################################################################################

def valmap(func, d, factory=dict):
    return factory(**{k: func(v) for k, v in d.items()})


def keymap(func, d, factory=dict):
    return factory(**{func(k): v for k, v in d.items()})


def keyfilter(func, d, factory=dict):
    return factory(**{k: v for k, v in d.items() if func(k)})


# Empty and Reserved - representing an unassigned variable #########################################


Empty = inspect.Parameter.empty  # unassigned parameter that should be assigned


class Missing:
    """A sentinel object representing missing fields in dataclasses.
    Using `Empty` causes errors in dataclasses because assigning a default value
    in dataclasses is equal to not assigning anything.
    """
    pass


def is_empty(value):
    return value in [Empty, inspect.Parameter.empty]


class Reserved:  # placeholder to mark parameters that shouldn't be assigned / are reserved
    @staticmethod
    def partial(func, **kwargs):
        """Applies partial to func only if all supplied arguments are Reserved."""
        for k, v in kwargs.items():
            if default_args(func)[k] is not Reserved:
                raise ValueError(
                    f"The argument {k} should be reserved in order to be assigned a value."
                    + " The reserved argument might have been overridden with partial.")
        return partial(func, **kwargs)

    @staticmethod
    def call(func, **kwargs):
        """Calls func only if all supplied arguments are Reserved."""
        return Reserved.partial(func, **kwargs)()
