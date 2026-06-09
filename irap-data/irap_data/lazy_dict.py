"""A minimal mapping with attribute access and lazy fields.

Used as the container for `Dataset.info` so that:
- consumers can access fields with either ``info['x']`` or ``info.x``;
- expensive info entries (e.g. disk-cached statistics) can be wrapped in
  ``Lazy(...)`` and computed only on first access.
"""

from collections import abc
import typing as T


class _NoValue:
    pass


class Lazy:
    """Wraps a zero-argument callable, caching its result on first call."""

    __slots__ = ("_get", "_value")

    def __init__(self, get: T.Callable[[], T.Any]):
        self._get = get
        self._value = _NoValue

    def __call__(self):
        if self._value is _NoValue:
            self._value = self._get()
        return self._value


class LazyDict(abc.MutableMapping):
    """A dict-like container with attribute access and lazy fields.

    ``Lazy`` values are evaluated on first access and the result replaces the
    wrapper in the backing dict. Iteration, ``len``, ``in``, ``keys``,
    ``values``, ``items`` follow the ``MutableMapping`` contract over keys.
    """

    __slots__ = ("_dict",)

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(
                f"LazyDict expects at most one positional argument, got {len(args)}.")
        backing: dict = dict(args[0]) if args else dict()
        backing.update(kwargs)
        object.__setattr__(self, "_dict", backing)

    def __getitem__(self, key):
        val = self._dict[key]
        if isinstance(val, Lazy):
            val = self._dict[key] = val()
        return val

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, key):
        try:
            val = self._dict[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, Lazy):
            val = self._dict[key] = val()
        return val

    def __setattr__(self, key, value):
        self._dict[key] = value

    def __repr__(self):
        parts = [f"{k}=<unevaluated>" if isinstance(v, Lazy) else f"{k}={v!r}"
                 for k, v in self._dict.items()]
        return f"LazyDict({', '.join(parts)})"
