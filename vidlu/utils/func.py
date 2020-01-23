import inspect
from collections.abc import Mapping
from inspect import signature
from functools import partialmethod, partial, reduce, wraps
import itertools
import typing

from vidlu.utils import text
from .collections import NameDict
from vidlu.utils import tree, misc


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


class frozen_partial(partial):
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
    return frozen_partial(func, **{k: v for k, v in default_args(func) if v is not Empty})


def tryable(func, default_value, error_type=Exception):
    def try_(*args, **kwargs):
        # noinspection PyBroadException
        try:
            return func(*args, **kwargs)
        except error_type:
            return default_value

    return try_


def _dummy(*a, **k):
    pass


class _FuncTree(partial, Mapping):
    def __new__(cls, *args, **kwargs):
        func, *args = args
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
    return _FuncTree(func, *args, **kwargs)


def functree_shallow(func, *args, **kwargs):
    return _FuncTree(func, *args, **{**default_args(func), **kwargs})


def call_with_args_from_dict(func, dict_):
    par = params(func)
    return func(**misc.update_existing_items(
        {k: v for k, v in par.items() if v is not Empty or k in dict_}, dict_))


def partial_with_args_from_dict(func, dict_):
    par = params(func)
    return partial(func, **misc.update_existing_items(
        {k: v for k, v in par.items() if v is not Empty or k in dict_}, dict_))


# parameters/arguments #############################################################################

def param_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return len(signature(func).parameters)


def default_args(func) -> NameDict:
    return NameDict({k: v.default for k, v in signature(func).parameters.items()
                     if v.default is not Empty})


def params(func) -> NameDict:
    if not callable(func):
        raise TypeError("The provided type is not callable.")
    try:
        return NameDict({k: v.default for k, v in signature(func).parameters.items()})
    except ValueError as ex:
        if 'no signature found for builtin' not in str(ex):
            raise
        return NameDict()


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

    def copy(self):
        return ArgTree({k: v.copy() if isinstance(v, ArgTree) else v for k, v in self.items()})

    @staticmethod
    def from_func(func):
        return ArgTree(
            {k: ArgTree.from_func(v) if callable(v) and v not in (Empty, Reserved) else v
             for k, v in params(func).items()})


class FrozenArgTree(ArgTree):
    """An argtree whose arguments cannot be overridden."""


class EscapedArgTree:
    __slots__ = ('argtree',)

    def __init__(self, argtree):
        self.argtree = argtree


def argtree_partial(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) != 1:
        raise ValueError("The arguments should be either a single positional argument"
                         + " or 1 or more keyword arguments.")
    # TODO: do not mutate arguments
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, FrozenArgTree):
            tree[k] = argtree_frozen_partial(default_args(func)[k], v)
        if isinstance(v, ArgTree):
            tree[k] = argtree_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    if isinstance(tree, FrozenArgTree):
        return frozen_partial(func, **tree)
    return partial(func, **tree)


def argtree_frozen_partial(func, *args, **kwargs):
    if len(args) + int(len(kwargs) > 0) > 1:
        raise ValueError("The arguments should be either a single positional argument"
                         + " or 0 or more keyword arguments.")
    tree = args[0] if len(args) == 1 else kwargs
    for k, v in list(tree.items()):
        if isinstance(v, ArgTree):
            tree[k] = argtree_frozen_partial(default_args(func)[k], v)
        elif isinstance(v, EscapedArgTree):
            tree[k] = v.argtree
    return frozen_partial(func, **tree)


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
        dargs = params(func)
        for k, v in kwargs.items():
            if dargs[k] is not Reserved:
                raise ValueError(
                    f"The argument {k} should be reserved in order to be assigned a value."
                    + " The reserved argument might have been overridden with partial.")
        return partial(func, **kwargs)

    @staticmethod
    def call(func, **kwargs):
        """Calls func only if all supplied arguments are Reserved."""
        return Reserved.partial(func, **kwargs)()


def make_multiinput(func):
    @wraps(func)
    def wrapper(x, *a, **k):
        if isinstance(x, tuple):
            return tuple(func(x_, *a, **k) for x_ in x)
        return func(x, *a, **k)

    return wrapper


def multiinput_method(func):
    @wraps(func)
    def wrapper(self, x, *a, **k):
        if isinstance(x, tuple):
            return tuple(func(self, x_, *a, **k) for x_ in x)
        return func(self, x, *a, **k)

    return wrapper


def func_to_class(func, call_params_count=1, *, superclasses=(), method_name='__call__',
                  name=None, ):
    from inspect import signature, Parameter
    name = name or text.to_pascal_case(func.__name__)  # PascalCase
    pnames = list(signature(func).parameters.keys())
    if not isinstance(superclasses, typing.Sequence):
        superclasses = (superclasses,)

    call_pnames = pnames[0:call_params_count]
    init_pnames = pnames[call_params_count:]

    exec(f"""
class {name}(*superclasses):
    func=func
    def __init__(self, {', '.join(init_pnames)}):
        super().__init__()
        self.args=dict({', '.join(k + '=' + k for k in init_pnames)})
    def {method_name}(self, {', '.join(call_pnames)}):
        return func({', '.join(call_pnames)}, **self.args)""",
         {'func': func}, locals())
    class_ = locals()[name]
    defargs = default_args(func)
    defaults = tuple(defargs.values())  # works for decorated procedures
    if tuple(defargs.keys()) != tuple(init_pnames[len(init_pnames) - len(defargs):]):
        raise NotImplementedError("Cannot convert a functools.partial into a class "
                                  + "if there are unassigned arguments after assigned arguments.\n"
                                  + f"The signature is {params(func)}")
    if defaults:
        split = max(0, len(defaults) - len(init_pnames))  # avoiding split = -len(init_pnames)
        class_.__init__.__defaults__ = defaults[split:]  # rightmost arguments go to __init__
        class_.__call__.__defaults__ = defaults[0:split]  # not to get [0:-0] with -len(init_pnames)
    class_.__module__ = func.__module__
    return class_


def class_to_func(class_, name=None):
    if len(default_args(class_.__call__)) > 0:
        raise ValueError("The `__call__` method of the class should have no default arguments.")
    from inspect import signature, Parameter
    name = name or text.to_snake_case(class_.__name__)
    init_pnames = list(signature(class_).parameters.keys())
    call_pnames = list(signature(class_.__call__).parameters.keys())[1:]
    pnames = call_pnames + init_pnames

    exec(f"""
def {name}({', '.join(pnames)}):
    return class_({', '.join(init_pnames)})({', '.join(call_pnames)})""",
         {'class_': class_}, locals())
    func = locals()[name]
    func.__defaults__ = class_.__init__.__defaults__
    if class_.__init__.__defaults__ is None and len(default_args(class_.__init__)) > 0:
        raise NotImplementedError("Not implemented for decorated __init__.")
    func.__module__ = class_.__module__
    return func


def type_checked(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        ba = dict(**ba.arguments, args=ba.args[len(ba.arguments):], kwargs=ba.kwargs)
        fail = False
        for name, type in func.__annotations__.items():
            type_origin = typing.get_origin(type)
            if type_origin is not None:
                if type_origin is typing.Literal:
                    if ba[name] not in typing.get_args(type):
                        fail = True
            if fail or not isinstance(ba[name], type):
                val_str = str(ba[name])
                val_str = val_str if val_str < 80 else val_str[:77] + '...'
                raise TypeError(f"The argument {name}={val_str} is not of type {type}.")
        return func(*args, **kwargs)

    return wrapper
