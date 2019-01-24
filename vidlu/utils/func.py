import inspect
from inspect import signature
from functools import partialmethod, partial, reduce, wraps

from .collections import NamespaceDict
from .misc import tree_to_paths, Empty


# Wrappers #########################################################################################

class hard_partial(partial):
    """
    Like partial, but doesn't allow changing already chosen keyword arguments.
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
    return lambda x: reduce(lambda a, f: f(a), funcs, initial=x)


# TreeArgs, treeargs_partial #######################################################################

class ArgTree(NamespaceDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EscapedArgTree:
    def __init__(self, argtree):
        self.argtree = argtree


def argtree_partial(func, **kwargs):
    args = kwargs
    for k, v in list(args.items()):
        if isinstance(v, ArgTree):
            args[k] = argtree_partial(default_args(func)[k], **v)
        elif isinstance(v, EscapedArgTree):
            args[k] = v.argtree
    return partial(func, **args)


def argtree_partialmethod(func, **kwargs):
    args = kwargs
    for k, v in list(args.items()):
        if isinstance(v, ArgTree):
            args[k] = argtree_partial(default_args(func)[k], **v)
        elif isinstance(v, EscapedArgTree):
            args[k] = v.argtree
    return partialmethod(func, **args)


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


def default_args(func) -> NamespaceDict:
    return NamespaceDict(**{k: v.default for k, v in signature(func).parameters.items()
                            if v.default is not Empty})


def params(func) -> NamespaceDict:
    return NamespaceDict(**{k: v.default for k, v in signature(func).parameters.items()})


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
    return found if return_tree else tree_to_paths(found)


def dot_default_args(fun):
    if inspect.isclass(fun):
        kwargs = default_args(fun.__init__)
    elif callable(fun):
        kwargs = default_args(fun)
    else:
        assert False
    for k, v in kwargs.items():
        assert not hasattr(fun, k)
        setattr(fun, k, v)
    return fun
