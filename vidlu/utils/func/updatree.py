import copy
import itertools
import sys
import typing as T

from vidlu.utils.collections import NameDict
from ._func import partial, params, default_args, is_empty


# FuncTree #########################################################################################


class UpdatreeMixin:
    def apply(self, arg):
        raise NotImplementedError()


class UpdatreeBase(UpdatreeMixin, NameDict):
    pass


class UpdatreeLeaf(UpdatreeMixin):
    """Node representing a value, but, since it inherits UpdatreeMixin, it
    requires an attribute with its name (defined in the parent) to already exist."""

    def __init__(self, value):
        self.value = value

    def apply(self, x):
        return self.value


class EscapedItem:
    __slots__ = ('item',)

    def __init__(self, item):
        self.item = item

    def __call__(self):
        return self.item


def nofunc(*a, **k):  # value marking that there is no function in a FuncTree node
    raise RuntimeError("nofunc called. Probably a node in a FuncTree has no function.")
    pass


def _extract_func_and_kwargs_for_functree(*funcs, **kwargs):
    func = nofunc
    kw = dict()
    for f in funcs:
        if isinstance(f, FuncTree):
            kw.update(f.keywords)
            f = f.func
        if f is not nofunc:
            func = f
    kw.update(kwargs)
    return func, kw


class FuncTree(partial, UpdatreeBase):
    nofunc = nofunc

    def __new__(cls, *args, **kwargs):
        func, kw = _extract_func_and_kwargs_for_functree(*args, **kwargs)
        obj = partial.__new__(cls, func, **kw)
        obj._func = obj.func
        return obj

    def __getattr__(self, key):
        return self.keywords[key] if key in self.keywords else partial.__getattribute__(self, key)

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

    def apply(self, func) -> callable:
        if isinstance(func, FuncTree):
            return FuncTree(func, self)
        else:
            kwargs = {k: v.apply(params(func)[k]) if isinstance(v, UpdatreeMixin) else v
                      for k, v in self.keywords.items()}
            return partial(func, **kwargs)

    def keys(self):
        return self.keywords.keys()

    def values(self):
        return self.keywords.values()

    def items(self):
        return self.keywords.items()

    def update(self, *args, **kwargs):
        func, kw = _extract_func_and_kwargs_for_functree(*args, **kwargs)
        if func is not nofunc:
            self.func = func  # error (read_only attribute)
        for k, v in {**kw, **kwargs}.items():
            if isinstance(v, FuncTree):
                if k not in self or not callable(self[k]):
                    self[k] = v.copy()
                else:
                    if not isinstance(self[k], FuncTree):
                        self[k] = FuncTree(self[k])
                    self[k].update(v)
            elif isinstance(v, EscapedItem):
                self[k] = v()
            else:
                self[k] = v

    def copy(self):
        return FuncTree(self.func, **{k: v.copy() if isinstance(v, FuncTree) else v
                                      for k, v in self.items()})


def functree_from_func(func, kwargs=None, extract_kwargs=False, depth=sys.maxsize):
    if depth < 0:
        raise RuntimeError("Tree depth must be at least 0.")
    kwargs = kwargs or dict()
    if extract_kwargs:
        try:
            dargs = default_args(func)
            kwargs_iter = itertools.chain(dargs.items(), kwargs.items())
        except ValueError:
            kwargs_iter = kwargs.items()
    else:
        kwargs_iter = kwargs.items()
    if depth == 0:
        kwargs = {k: v for k, v in kwargs_iter}
    else:
        kwargs = {k: (functree_from_func(v, extract_kwargs=extract_kwargs, depth=depth - 1)
                      if callable(v) else v)
                  for k, v in kwargs_iter}
    return FuncTree(func, **kwargs)


def _check_item_existence(obj, k):
    return obj[k]


def _check_attr_existence(obj, k):
    return getattr(obj, k)


class StrictIndexableUpdatree(UpdatreeBase):
    _check = True

    def apply(self, obj):
        result = type(obj)(obj)
        for k, v in self.items():
            if self._check:
                _check_item_existence(obj, k)
            result[k] = v.apply(result[k]) if isinstance(v, UpdatreeMixin) else v
        return result


class IndexableUpdatree(StrictIndexableUpdatree):
    _check = False


class StrictObjectUpdatree(StrictIndexableUpdatree):
    _check = True

    def apply(self, obj):
        result = copy.copy(obj)
        for k, v in self.items():
            if self._check:
                _check_attr_existence(obj, k)
            setattr(result, k, v.apply(getattr(result, k)) if isinstance(v, UpdatreeMixin) else v)
        return result


class ObjectUpdatree(StrictObjectUpdatree):
    _check = False


class AppendUpdatree(UpdatreeBase):
    def apply(self, arg):
        result = type(arg)(arg)
        for k, v in self.items():
            result[k] = v.apply(result[k]) if isinstance(v, UpdatreeMixin) else v
        return result


class ArgTree(UpdatreeBase):
    class FUNC:
        pass

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            if ArgTree.FUNC not in kwargs:
                kwargs[ArgTree.FUNC] = args[0]
            args = args[1:]
        super().__init__(*args, **kwargs)

    def apply(self, func):
        return argtree_partial(func, self)

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
    def from_func(func, depth=sys.maxsize):
        return ArgTree(
            {k: ArgTree.from_func(v, depth - 1) \
                if callable(v) and not is_empty(v) and depth > 0 else v
             for k, v in params(func).items()})


def _process_argtree_partial_args(*args, **kwargs):
    func, *args = args if len(args) > 0 and callable(args[0]) else (nofunc,)
    if len(args) > 1:
        raise RuntimeError("Too many positional arguments. There should be"
                           + " at most 2 if the first one is a callable.")
    tree = args[0] if len(args) == 1 and len(kwargs) == 0 else dict(*args, **kwargs)
    if func is nofunc and (func := tree.pop(ArgTree.FUNC, nofunc)) is nofunc:
        raise ValueError("A function should be supplied either as args[0] or"
                         + f" args[0][{repr(ArgTree.FUNC)}] or args[1][{repr(ArgTree.FUNC)}]")
    return func, tree


def argtree_partial_(args, kwargs, partial_f=partial):
    func, tree = _process_argtree_partial_args(*args, **kwargs)

    par = None

    def get_param(k):
        nonlocal par
        if par is None:
            par = params(func)
        return par[k]

    kwargs = {
        k: argtree_partial_((default_args(func)[k],), v, partial_f=partial_f)
        if isinstance(v, ArgTree)
        else v.item if isinstance(v, EscapedItem)
        else v.apply(get_param(k)) if isinstance(v, UpdatreeMixin)
        else v
        for k, v in tree.items()}
    return partial_f(func, **kwargs)


def argtree_partial(*args, **kwargs):
    return argtree_partial_(args, kwargs, partial_f=partial)
