from collections.abc import Sequence, KeysView, ValuesView, ItemsView
from functools import reduce

import vidlu.utils.func as vuf


# Record

class _LazyField:
    __slots__ = "get"

    def __init__(self, get):
        if vuf.param_count(get) not in [0, 1]:
            raise ValueError("get should be a callable with either 0 (simple lazy evaluation) or 1"
                             + " parameter (in case of referring back to the 'Record' object).")
        self.get = get


class Record(Sequence):  # Sized, Iterable len, iter
    r"""
    An immutable sequence (supports numeric indexes) that behaves as a mapping
    as well (supports string keys) and supports the dot operator for accessing
    elements.

    A field of a record can be lazily evaluated. Such a field is represented
    with a function. The output of the function is cached when accessed for
    the first time.

    Example:
        >>> r = Record(a=2, b=53)
        Record(a=2, b=53)
        >>> r.a == r['a'] == r[0]
        True
        >>> list(r)
        [2, 53]
        >>> dict(r)
        {'a': 2, 'b': 53}
        >>> r = Record(a_=lambda: 2+3, b=7)
        Record(a=<unevaluated>, b=7)
        >>> r.a; print(r)
        Record(a=5, b=7)
        >>> a, y = r; print(a, y)
        5 7
        >>> print([(k, v) for k, v in r.items()])
        [('a', 5), ('b', 7])]
        >>> r2 = Record(b_=lambda: 4, c_=lambda: 8)
        Record(b=<unevaluated>, c=<unevaluated>)
        >>> r3 = Record(r2, c=1, d=2)
        Record(b=<unevaluated>, c=1, d=2)
        >>> r3[('d','b')]
        Record(d=2, b=<unevaluated>)
    """

    __slots__ = "_dict"

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("All arguments but the first one must be keyword arguments."
                             + " The optional positional argument can only be a Record or Mapping.")
        if len(args) == 1:
            d = args[0]
            dict_ = dict(d._dict if isinstance(d, Record) else d, **kwargs)
        else:
            dict_ = kwargs
        dict_ = dict((k[:-1], _LazyField(v)) if k.endswith('_') else (k, v)
                     for k, v in dict_.items())
        if not all(type(k) is str for k in dict_.keys()):
            raise ValueError("Record keys must be strings.")
        self._dict = dict_

    def __getattr__(self, key):  # TODO: add __dir__
        return self[key]

    def __getitem__(self, key):
        """
        :param key: int or str or List[str]
        :return:
        """
        if isinstance(key, Sequence) and not isinstance(key, str):
            return Record(dict([(k, self[k]) if self.is_evaluated(k) else (k + "_", lambda: self[k])
                                for k in key]))
        else:
            if isinstance(key, int):
                key = tuple(self.keys())[key]
            val = self._dict[key]
            if isinstance(val, _LazyField):
                val = self._dict[key] = val.get() if vuf.param_count(val.get) == 0 \
                    else val.get(self)
            return val

    def __contains__(self, item):
        """Only keys are compared. """
        return item in self._dict

    def __iter__(self):  # returns values, Iterable
        return iter(self.values())

    def __len__(self):  # Sized
        return len(self._dict)

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.items(), other.items()))

    def __getstate__(self):
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        self._dict = state

    def _to_string(self, to_string_func=repr):
        fields = ", ".join(
            [f"{k}={to_string_func(self[k])}" if self.is_evaluated(k) else f"{k}=<unevaluated>"
             for k in self.keys()])
        return f"Record({fields})"

    def __str__(self):
        return self._to_string(str)

    def __repr__(self):
        return self._to_string(repr)

    def evaluate(self):
        for k in self.keys():
            _ = self[k]

    def is_evaluated(self, key):
        return not isinstance(self._dict[key], _LazyField)

    def join(self, other: 'Record', *others, overwrite=False):
        if len(others) > 0:
            return reduce(lambda a, b: a.join(b, overwrite=overwrite), [self, other, *others])
        if not overwrite and any(a in self.keys() for a in other.keys()):
            raise ValueError("Sets of field names should be disjoint if `overwrite == False`.")
        return Record(self, **other._dict)

    def keys(self) -> KeysView:
        return self._dict.keys()

    def values(self):
        return RecordValuesView(self)

    def items(self):
        return RecordItemsView(self)


class RecordItemsView(Record):
    def __iter__(self):
        return ((k, self[k]) for k in self.keys())

    def __contains__(self, item):
        return any(v == item for k, v in self)


class RecordValuesView(Record):
    def __iter__(self):
        return (self[k] for k in self.keys())

    def __contains__(self, item):
        return item in self.keys()
