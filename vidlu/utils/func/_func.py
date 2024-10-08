import inspect
from inspect import signature
import functools
import typing as T
import warnings

from typeguard import typechecked

from vidlu.utils import text, tree
from vidlu.utils.collections import NameDict


def identity(x):
    return x


# ArgHolder ########################################################################################

class ArgHolder:
    __slots__ = "args", "kwargs"

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def bind(self, func):
        return Partial(func, *self.args, **self.kwargs)

    def call(self, func):
        return func(*self.args, **self.kwargs)

    def __repr__(self):
        args = ", ".join(repr(a) for a in self.args)
        return f"ArgHolder({args}, kwargs={repr(self.kwargs)})"

    def __str__(self):
        return repr(self)


# Partial ##########################################################################################

class Partial(functools.partial):
    """partial with a more informative error message and parameters accessible
    using the dot operator.

    Example::
    >>> p = partial(lambda a=5, b=7: print(a, b), a=1)
    >>> assert p.a == 1 and p.b == 7
    >>> p()
    1 7
    """

    def __new__(cls, func, /, *args, **keywords):
        """__new__ is redefined so that `Partial(functools.partial(foo, ...)).func == foo`"""
        if not callable(func):
            raise TypeError("the first argument must be callable")

        # partial.__new__ uses hasattr(func, func) instead, which would break
        if isinstance(func, functools.partial) and not isinstance(func, FrozenPartial):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        return super().__new__(cls, func, *args, **keywords)

    def __call__(self, *args, **kwargs):
        try:
            func = self.func
            sig = inspect.signature(func)
            sigparams = list(sig.parameters.items())
            if len(sigparams) > 0 and sigparams[-1][1].kind is not inspect.Parameter.VAR_KEYWORD:
                params_ = set(params(func).keys())
                provided = set({**self.keywords, **kwargs}.keys())
                unexpected = provided.difference(params_)
                if len(unexpected) > 0:
                    func_name = getattr(func, "__name__", str(func))
                    raise RuntimeError(f"{func_name} with signature {sig} got unexpected arguments"
                                       + f" {unexpected}.")
        except ValueError:
            pass
        return functools.partial.__call__(self, *args, **kwargs)

    def __getitem__(self, item):
        # Using `params(self)[item]` instead of the line below causes infinite recursion
        return self.keywords[item] if item in self.keywords else params(self.func)[item]

    def __getattr__(self, item):
        """Returns keyword assigned or default arguments.

        It does not work if some argument has the name "func", "args", or
        "keywrods". The indexing operator should be used in these cases.
        The "func" attribute returns the wrapped function, "args" returns the
        args attribute of the object (assigned positional arguments), and
        "keywords" returns asssigned keyword arguments."""
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(f"{e}") from e


partial = Partial


class FrozenPartial(functools.partial):
    """Like Partial, but doesn't allow changing already chosen keyword
    arguments.

    Although `Partial.__new__` looks like it should copy the `keywords`
    attribute, this somehow works too: `Partial(FrozenPartial(f, x=2), x=3)`
    """

    def __new__(cls, func, /, *args, **keywords):
        """__new__ is redefined so that `Partial(functools.partial(foo, ...)).func == foo`"""
        if not callable(func):
            raise TypeError("the first argument must be callable")

        return super().__new__(cls, func, *args, **keywords)

    def __call__(self, *args, **kwargs):
        for k in kwargs.keys():
            if k in self.keywords:
                raise RuntimeError(f"Parameter {k} is frozen and can't be overridden.")
        return Partial.__call__(self, *args, **kwargs)


def frozen_partial(*a, **k):
    warnings.warn("frozen_partial deprecated. Use FrozenPartial.")
    return FrozenPartial(*a, **k)


# Empty, Reserved and Required - representing an unassigned variable ###############################


# Object that can be used for marking required arguments that come after some keyword argument
Empty = inspect.Parameter.empty


class Required(Empty):
    """Object for marking fields in dataclasses as required arguments when the
    they come after fields with default values.

    `Empty` (`inspect.Parameter.empty`) cannot be used because it causes an
    error saying that a non-default argument follows a default argument.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError('`Required` constructor has been called, indicating that a parameter with'
                        + ' default value `Required` is not assigned a "real" value.')


class Reserved(Empty):  # marker for parameters that shouldn't be assigned / are reserved
    @staticmethod
    def partial(func, **kwargs):
        """Applies partial to func only if all supplied arguments are Reserved."""
        dargs = params(func)
        for k, v in kwargs.items():
            if dargs[k] is not Reserved:
                warnings.warn(
                    f"The argument {k} is assigned although it should be marked `Reserved`."
                    + " The reserved argument might have been overridden with partial.")
        return Partial(func, **kwargs)

    @staticmethod
    def call(func, **kwargs):
        """Calls func only if all supplied arguments are Reserved."""
        return Reserved.partial(func, **kwargs)()


def is_empty(arg):
    return isinstance(arg, type) and issubclass(arg, Empty)


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

    @functools.wraps(func0)
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


def tryable(func, default_value, error_type=Exception):
    def try_(*args, **kwargs):
        # noinspection PyBroadException
        try:
            return func(*args, **kwargs)
        except error_type:
            return default_value

    return try_


# Assignable arguments picking #####################################################################

def _no_assignment_cond(k, v, default):
    return True


TKwargsPolicy = T.Literal["error", "pick", "ignore"]


@typechecked
def pick_assignable_args(func, args_dict, return_other=False, assignment_cond=None,
                         kwargs_policy: TKwargsPolicy = "error"):
    if assignment_cond is None:
        assignment_cond = _no_assignment_cond
    parameters = list(signature(func).parameters.items())
    if len(parameters) == 0:
        return (dict(), dict(args_dict)) if return_other else dict()
    elif parameters[-1][1].kind == inspect.Parameter.VAR_KEYWORD:
        if kwargs_policy == "error":
            raise ValueError(f"func accepts unspecified keyword arguments, but {kwargs_policy=}"
                             + f" does not allow it")
        elif kwargs_policy == "pick_all":
            return ({**args_dict}, dict()) if return_other else args_dict
    assignable = {k: args_dict[k] for k, v in parameters
                  if k in args_dict and assignment_cond(k, args_dict[k], v.default)}
    if return_other:
        other = {k: v for k, v in args_dict.items() if k not in assignable}
        return assignable, other
    return assignable


def call_with_assignable_args(func, args_dict, return_other=False, assignment_cond=None,
                              kwargs_policy: TKwargsPolicy = "error"):
    a = pick_assignable_args(func, args_dict, return_other=return_other,
                             assignment_cond=assignment_cond, kwargs_policy=kwargs_policy)
    return (func(**a[0]), a[1]) if return_other else func(**a)


def arg_picking(func, return_other=False, assignment_cond=None,
                kwargs_policy: TKwargsPolicy = "error"):
    @functools.wraps(func)
    def wrapper(**ctx):
        return call_with_assignable_args(func, ctx, return_other=return_other,
                                         assignment_cond=assignment_cond,
                                         kwargs_policy=kwargs_policy)

    return wrapper


# parameters/arguments #############################################################################

def param_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return len(signature(func).parameters)


def positional_param_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return sum(1 for param in signature(func).parameters.values() if
               param.kind == param.POSITIONAL_OR_KEYWORD)


def default_args(func) -> NameDict:
    return NameDict({k: v.default for k, v in signature(func).parameters.items()
                     if v.default is not Empty})


def params(func) -> NameDict:
    if not callable(func):
        raise TypeError("The provided type is not callable.")
    try:
        return NameDict({k: v.default for k, v in signature(func).parameters.items()})
    except ValueError as e:
        if 'no signature found for builtin' not in str(e):
            raise
        return NameDict()


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
    from .updatree import ArgTree
    found = ArgTree(found)
    return found.items() if return_tree else tree.flatten(found)


def inherit_missing_args(parent_function):
    parent_default_args = default_args(parent_function)

    def decorator(func):
        inherited_args = {k: parent_default_args[k] for k, v in params(func).items()
                          if v is Empty and k in parent_default_args}
        return Partial(func, **inherited_args)

    return decorator


# Dict map, filter #################################################################################

def valmap(func, d, factory=dict):
    return factory(**{k: func(v) for k, v in d.items()})


def keymap(func, d, factory=dict):
    return factory(**{func(k): v for k, v in d.items()})


def keyfilter(func, d, factory=dict):
    return factory(**{k: v for k, v in d.items() if func(k)})


# Decorators #######################################################################################

def vectorize(func=None, /, *, n=1):
    """

    Args:
        func (Callable): a function that we want to apply in parallel to batched parameters.
        n (int): number of arguments of different types

    Producess a wrapper around f=func that computes as follows:
        wrapper(func)((a11, .., a1n), .., (am1, ..., amn), b1, .., bk) =
            (func(a11, .., am1, b1, .., bk), .., func(1n, .., amn, b1, .., bk))

    Returns:
        wrapper(func) if func is provided, and wrapper otherwise.
    """

    def wrap(func):
        @functools.wraps(func)
        def wrapper(*a, **k):
            if isinstance(a[0], tuple):
                vector, a_shared = a[:n], a[n:]
                return tuple(func(*vec, *a_shared, **k) for vec in zip(*vector))
            return func(*a, **k)

        return wrapper

    if func is None:
        return wrap
    return wrap(func)


# Caching ##########################################################################################

class _FactoryHolder:
    __slots__ = 'factory'

    def __init__(self, factory):
        self.factory = factory


class Cached:
    """Wraps a function with no parameters. When called the first time, calls the function and
    stores its result and discards the function. When called again, returns the stored result."""

    __slots__ = '_item'

    def __init__(self, factory):
        self._item = _FactoryHolder(factory)

    def __call__(self):
        if isinstance(self._item, _FactoryHolder):
            self._item = self._item.factory()
        return self._item


# Other ############################################################################################

def func_to_class(func, call_params_count=1, *, superclasses=(), method_name='__call__', name=None):
    from inspect import signature
    name = name or text.to_pascal_case(func.__name__)  # PascalCase
    pnames = list(signature(func).parameters.keys())
    if not isinstance(superclasses, T.Sequence):
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
