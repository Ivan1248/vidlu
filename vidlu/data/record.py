from collections import abc
from functools import reduce
import typing as T

import vidlu.utils.func as vuf


# Record


class LazyField:
    __slots__ = "_get", "_value"

    def __init__(self, get):
        if vuf.param_count(get) not in [0, 1]:
            raise ValueError("get should be a callable with either 0 (simple lazy evaluation) or 1"
                             + " parameter (in case of referring back to the 'Record' object).")
        self._get = get
        self._value = None

    def __call__(self, record):
        """Although the Record instance discards the LazyField instance after calling it,
        we cache the value in case some other Record instance uses it too."""
        if self._value is None:
            self._value = self._get() if vuf.param_count(self._get) == 0 else self._get(record)
        return self._value


def _process_lazy_args(args: dict):
    return dict((k[:-1], LazyField(v)) if k.endswith('_') else (k, v) for k, v in args.items())


class Record(abc.Sequence):  # Sized, Iterable len, iter
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
        dict_ = _process_lazy_args(kwargs)
        if len(args) == 1:
            d = args[0]
            dict_ = dict(d._dict if isinstance(d, Record) else _process_lazy_args(dict(d)), **dict_)
        self._dict = dict_
        if any(isinstance(k, int) for k in dict_.keys()):
            raise ValueError("Record keys must be non-ints.")

    def __getitem__(self, key: T.Union[int, str, T.Sequence[int], T.Sequence[str]]):
        if isinstance(key, slice):
            return self[list(range(*key.indices(len(self))))]
        elif isinstance(key, abc.Sequence) and not isinstance(key, str):
            if len(key) == 0:
                return Record()
            if isinstance(key[0], int):
                all_keys = tuple(self.keys())
                key = [all_keys[i] for i in key]
            return Record({k: self._dict[k] for k in key})
        else:
            if isinstance(key, int):
                key = tuple(self.keys())[key]
            elif key not in self._dict:
                raise KeyError(
                    f'{repr(key)} is not among available keys: {", ".join(repr(k) for k in self._dict.keys())}.')
            val = self._dict[key]
            if isinstance(val, LazyField):
                val = self._dict[key] = val(self)
            return val

    def __len__(self):  # Sized
        return len(self._dict)

    def __contains__(self, item):  # Mapping
        raise TypeError("`in` operator not supported. Use the `values` or the `keys` method.")

    def __iter__(self):  # Sequence (returns values)
        # raise TypeError("`in` operator not supported. Use the `values` or the `keys` method.")
        return iter(self.values())

    def __getattr__(self, key):  # namespace
        return self[key]

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.items(), other.items()))

    def __getstate__(self):  # evaluates everything
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        self._dict = state

    def _to_string(self, func=repr):
        fields = ", ".join(
            [f"{k}={func(self[k])}" if type(self).is_evaluated(self, k) else f"{k}=<unevaluated>"
             for k in self.keys()])
        return f"{type(self).__name__}({fields})"

    def __str__(self):
        return self._to_string(str)

    def __repr__(self):
        return self._to_string(repr)

    def evaluate(self):
        for k in self.keys():
            _ = self[k]

    def is_evaluated(self, key):
        return not isinstance(self._dict[key], LazyField)

    def join(self, other: 'Record', *others, overwrite=False):
        if len(others) > 0:
            return reduce(lambda a, b: a.join(b, overwrite=overwrite), [self, other, *others])
        if not overwrite and any(a in self.keys() for a in other.keys()):
            raise ValueError("Sets of field names should be disjoint if `overwrite == False`.")
        return Record(self, **other._dict)

    def keys(self):  # Mapping
        return self._dict.keys()

    def values(self):  # Mapping
        return ListRecord(self)  # TODO

    def items(self):  # Mapping
        return RecordItemsView(self)  # TODO


def arrange(r, field_names):
    return Record({**r[field_names]._dict, **r._dict})


class RecordView(Record):
    def __init__(self, record):
        super().__init__()
        self._dict = record._dict


class DictRecord(RecordView):
    def __iter__(self):
        return iter(self._dict.keys())

    def __contains__(self, item):
        return any(v == item for v in self)

    def _to_string(self, func=repr):
        fields = ", ".join([k for k in self])
        return f"{type(self).__name__}({fields})"


class ListRecord(RecordView):
    def __iter__(self):
        return (self[k] for k in self._dict.keys())

    def __contains__(self, item):
        return any(v == item for v in self)

    def _to_string(self, func=repr):
        fields = ", ".join(
            [f"{func(self[k])}" if type(self).is_evaluated(self, k) else f"<unevaluated>"
             for k in self])
        return f"{type(self).__name__}({fields})"


class RecordItemsView(RecordView):
    def __iter__(self):
        return ((k, self[k]) for k in self._dict.keys())

    def __contains__(self, item):
        return any(v == item for v in self)
