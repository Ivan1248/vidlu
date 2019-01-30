import inspect
from inspect import signature
from functools import partialmethod, partial, reduce, wraps

from .collections import NameDict
from .misc import tree_to_paths


# Wrappers #########################################################################################

class hard_partial(partial):
    """
    Like partial, but doesn't allow changing already chosen keyword arguments.
    Strangely, works as desired, even though `partial.__new__` looks like it
    should copy the `keywords` attribute in case partial(hard_partial(f, x=2), x=3
    """

    def __call__(self, *args, **kwargs):
        for k in kwargs.keys():
            if k in self.keywords:
                raise ValueError(f"Parameter {k} is frozen and can't be overridden.")
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


# composition ######################################################################################

def chain(funcs):
    def chain_(x):
        for f in funcs:
            x = f(x)
        return x

    return chain_


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


class HardArgTree(ArgTree):
    pass


class EscapedArgTree:
    __slots__ = ('argtree',)

    def __init__(self, argtree):
        self.argtree = argtree


def argtree_partial(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument " +
                         "or one or more keyword arguments.")
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
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument " +
                         "or one or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, ArgTree):
            tree[k] = argtree_hard_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    return hard_partial(func, **tree)


def argtree_partialmethod(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument " +
                         "or one or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, ArgTree):
            tree[k] = argtree_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    return partialmethod(func, **tree)


# find_empty_args ##################################################################################

def find_empty_args(func):
    incar = []  # list of lists of string (list of paths)
    for k, v in params(func).items():
        if v is Empty:
            incar.append([k])
        elif callable(v):
            incar += [[k] + x for x in find_empty_args(v)]
    return incar


# parameters/arguments #############################################################################

def parameter_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return len(signature(func).parameters)


def default_args(func) -> NameDict:
    return NameDict(**{k: v.default for k, v in signature(func).parameters.items()
                       if v.default is not Empty})


def params(func) -> NameDict:
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

def dict_map(func, dict_):
    return {k: func(v) for k, v in dict_.items()}


def dict_filter(func, dict_):
    return {k: v for k, v in dict_.items() if func(v)}


# Empty and Full - representing an unassigned variable #############################################


Empty = inspect.Parameter.empty  # unassigned parameter that should be assigned


class Full:  # placeholder to mark parameters that shouldn't be assigned / are reserved
    pass
