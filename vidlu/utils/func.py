import inspect
from inspect import signature
from functools import partialmethod, partial, reduce, wraps
from toolz import valmap

from .collections import NameDict
from vidlu.utils.tree import tree_to_paths


# Wrappers #########################################################################################


class composition:
    __slots__ = 'funcs'

    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x


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


# ArgTree, argtree_partial #########################################################################

class ArgTree(NameDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, other):
        for k, v in other.items():
            if k not in self:
                self[k] = v
            if isinstance(self[k], ArgTree) and isinstance(k, ArgTree):
                self[k].update(v)
            else:
                self[k] = v

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


def params_deep(func):
    args = []  # list of lists of string (list of paths)
    for k, v in params(func).items():
        if callable(v):
            args += [[k] + x for x in ArgTree.from_func(v)]
        else:
            args.append([k])
    return args


def find_empty_params_tree(func):
    tree = dict()
    for k, v in params(func).items():
        if callable(v):
            subtree = find_empty_params_tree(v)
            if len(subtree) > 0:
                tree[k] = subtree
        elif v is Empty:
            tree[k] = v
    return ArgTree(tree)


def find_empty_params_deep(func):
    args = []  # list of lists of string (list of paths)
    for k, v in params(func).items():
        if v is Empty:
            args.append([k])
        elif callable(v):
            args += [[k] + x for x in find_empty_params_deep(v)]
    return args


# parameters/arguments #############################################################################

def parameter_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return len(signature(func).parameters)


def default_args(func) -> NameDict:
    return NameDict(**{k: v.default for k, v in signature(func).parameters.items()
                       if v.default is not Empty})


def params(func) -> NameDict:
    if type(func).__name__ == 'DataLoader' or isinstance(func, partial) and type(
            func.func).__name__ == 'DataLoader':
        pass
    return NameDict(**{k: v.default for k, v in signature(func).parameters.items()})


def find_default_arg(func, name_or_path, recursive=True, return_tree=False):
    path = [name_or_path] if isinstance(name_or_path, str) else name_or_path
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
    return found if return_tree else tree_to_paths(found)


def dot_default_args(func):
    for k, v in default_args(func).items():
        assert not hasattr(func, k)
        setattr(func, k, v)
    return func


# Dict map, filter #################################################################################

def valmap(func, d, factory=dict):
    return factory(**{k: func(v) for k, v in d.items()})


def keymap(func, d, factory=dict):
    return factory(**{func(k): v for k, v in d.items()})


def keyfilter(func, d, factory=dict):
    return factory(**{k: v for k, v in d.items() if func(k)})


# Empty and Reserved - representing an unassigned variable #########################################


Empty = inspect.Parameter.empty  # unassigned parameter that should be assigned


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
