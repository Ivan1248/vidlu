import inspect
from collections.abc import MutableMapping
from inspect import signature
from functools import partialmethod, partial, wraps, update_wrapper
import itertools
import typing
import warnings
import typing as T

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
    Even though `partial.__new__` looks like it should copy the `keywords`
    attribute, this somehow works too: `partial(frozen_partial(f, x=2), x=3)`
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


def find_empty_params_deep(func):
    return find_params_deep(func, predicate=lambda k, v: v is Empty)


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


def inherit_missing_args(parent_function):
    parent_default_args = default_args(parent_function)

    def decorator(func):
        inherited_args = {k: parent_default_args[k] for k, v in params(func).items()
                          if v is Empty and k in parent_default_args}
        return partial(func, **inherited_args)

    return decorator


# FuncTree #########################################################################################

class _NoFunc:
    def __call__(self, *args, **kwargs):
        pass


nofunc = _NoFunc()


class FuncTree(partial, MutableMapping):
    def __new__(cls, *func, **kwargs):
        if len(func) > 1:
            raise ValueError("At most 1 positional argument (a callable) is acceptable.")
        return partial.__new__(cls, nofunc if len(func) == 0 else func[0], **kwargs)

    def __getattr__(self, key):
        return self.keywords[key] if key in self.keywords else super().__getattr__(key)

    def __delitem__(self, key):
        del self.keywords[key]

    def __setitem__(self, key, val):
        self.keywords[key] = val

    def __eq__(self, other):
        if not isinstance(other, FuncTree):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.keywords

    def __getitem__(self, key):
        return self.keywords[key]

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

    def update(self, other, **kwargs):
        if isinstance(other, FuncTree) and other.func is not nofunc:
            self.func = other.func
        self.keywords.update(other, **kwargs)

    def update_deep(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 positional argument, got {len(args)}.")
        args = args[0] if len(args) > 0 else {}
        for k, v in {**args, **kwargs}.items():
            if k not in self:
                self[k] = v
            if isinstance(self[k], FuncTree) and isinstance(v, FuncTree):
                self[k].update_deep(v)
                if v.func is not nofunc:
                    self.func = v.func
            elif callable(self[k]) and isinstance(v, FuncTree):
                self[k] = FuncTree(self[k])
                self[k].update_deep(v)
            else:
                self[k] = v

    def copy(self):
        return FuncTree(self.func, {k: v.copy() if isinstance(v, FuncTree) else v
                                    for k, v in self.items()})

    def pop(self, key):
        self.keywords.pop(key)

    def popitem(self):
        self.keywords.popitem()

    def setdefault(self, key, val=None):
        self.keywords.setdefault(key, val)


def functree(func, **kwargs):
    kwargs = {k: functree(v) if callable(v) else v
              for k, v in itertools.chain(default_args(func).items(), kwargs.items())}
    return FuncTree(func, **kwargs)


def functree_shallow(func, **kwargs):
    return FuncTree(func, **{**default_args(func), **kwargs})


# ArgTree, argtree_partial #########################################################################

class ArgTree(NameDict):
    FUNC = ''

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            if ArgTree.FUNC not in kwargs:
                kwargs[ArgTree.FUNC] = args[0]
            args = args[1:]
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 positional argument, got {len(args)}.")
        args = args[0] if len(args) > 0 else {}
        for k, v in {**args, **kwargs}.items():
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


def _process_argtree_partial_args(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        func, *args = args
    else:
        func = nofunc
    if len(args) > 1:
        raise RuntimeError("Too many positional arguments. There should be"
                           + " at most 2 if the first one is a callable.")
    tree = args[0] if len(args) == 1 and len(kwargs) == 0 else dict(*args, **kwargs)
    if func is nofunc and (func := tree.pop(ArgTree.FUNC, nofunc)) is nofunc:
        raise ValueError("A function should be supplied either as args[0] or"
                         + f" args[0][{repr(ArgTree.FUNC)}] or args[1][{repr(ArgTree.FUNC)}]")
    return func, tree


def _argtree_partial(args, kwargs, partial_f=partial):
    func, tree = _process_argtree_partial_args(*args, **kwargs)
    kwargs = {
        k: _argtree_partial((default_args(func)[k],), v, partial_f=partial_f)
        if isinstance(v, ArgTree)
        else v.argtree if isinstance(v, EscapedArgTree)
        else v
        for k, v in tree.items()}
    return partial_f(func, **kwargs)


def argtree_partial(*args, **kwargs):
    return _argtree_partial(args, kwargs, partial_f=partial)


def argtree_frozen_partial(func, *args, **kwargs):
    return _argtree_partial(args, kwargs, partial_f=frozen_partial)


def find_params_argtree(func, predicate):
    tree = dict()
    for k, v in params(func).items():
        if callable(v):
            subtree = find_params_argtree(v, predicate)
            if len(subtree) > 0:
                tree[k] = subtree
        elif predicate(k, v):
            tree[k] = v
    return ArgTree(tree)


def find_empty_params_argtree(func):
    return find_params_argtree(func, predicate=lambda k, v: v is Empty)


# Dict map, filter #################################################################################

def valmap(func, d, factory=dict):
    return factory(**{k: func(v) for k, v in d.items()})


def keymap(func, d, factory=dict):
    return factory(**{func(k): v for k, v in d.items()})


def keyfilter(func, d, factory=dict):
    return factory(**{k: v for k, v in d.items() if func(k)})


# Empty and Reserved - representing an unassigned variable #########################################


# Object that can be used for marking required arguments that come after some keyword argument
Empty = inspect.Parameter.empty


class Required:
    """Object for marking fields in dataclasses as required arguments when the
    they come after fileds with default values.
    
    `Empty` (`inspect.Parameter.empty`) cannot be used because it causes an 
    error saying that a non-default argument follows a default argument. 
    """

    def __init__(self):
        raise TypeError('`Required` constructor has been called, indicating that a parameter with'
                        + ' default value `Required` is not assigned a "real" value.')


class Reserved:  # marker for parameters that shouldn't be assigned / are reserved
    @staticmethod
    def partial(func, **kwargs):
        """Applies partial to func only if all supplied arguments are Reserved."""
        dargs = params(func)
        for k, v in kwargs.items():
            if dargs[k] is not Reserved:
                warnings.warn(
                    f"The argument {k} is assigned even though it should he marked `Reserved`."
                    + " The reserved argument might have been overridden with partial.")
        return partial(func, **kwargs)

    @staticmethod
    def call(func, **kwargs):
        """Calls func only if all supplied arguments are Reserved."""
        return Reserved.partial(func, **kwargs)()


# Other ############################################################################################

def vectorize(func):
    @wraps(func)
    def wrapper(x, *a, **k):
        if isinstance(x, tuple):
            return tuple(func(x_, *a, **k) for x_ in x)
        return func(x, *a, **k)

    return wrapper


def func_to_class(func, call_params_count=1, *, superclasses=(), method_name='__call__', name=None):
    from inspect import signature
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
    from inspect import signature
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
        for name, type_ in func.__annotations__.items():
            type_origin = typing.get_origin(type_)
            if type_origin is not None:
                if type_origin is typing.Literal:
                    if ba[name] not in typing.get_args(type_):
                        fail = True
            if fail or not isinstance(ba[name], type_):
                val_str = str(ba[name])
                val_str = val_str if len(val_str) < 80 else val_str[:77] + '...'
                raise TypeError(f"The argument {name}={val_str} is not of type {type_}.")
        return func(*args, **kwargs)

    return wrapper


def assign_available_args(func, args_dict, assignment_cond=None):
    if assignment_cond is None:
        assignment_cond = lambda k, v, default: True
    kwargs = {k: args_dict[k] for k, default in params(func)
              if k in args_dict and assignment_cond(k, args_dict[k], default)}
    return partial(func, **kwargs)
