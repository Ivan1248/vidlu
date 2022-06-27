import os
from abc import ABC
from collections import abc
from functools import reduce
import typing as T
import multiprocessing

import vidlu.utils.func as vuf


# Record

class _NoValue:
    pass


class LazyField:
    __slots__ = "_get", "_value"

    def __init__(self, get):
        if vuf.positional_param_count(get) not in [0, 1]:
            raise ValueError("get should be a callable with either 0 (simple lazy evaluation) or 1"
                             + "positional parameter (for referring back to the 'Record' object).")
        self._get = get
        self._value = _NoValue

    def __call__(self, record=None):
        """Although the Record instance discards the LazyField instance after calling it,
        we cache the value in case some other Record instance uses it too."""
        if self._value is _NoValue:
            self._value = self._get() if vuf.positional_param_count(self._get) == 0 else self._get(
                record)
        return self._value


def _process_lazy_arg(k, v):
    return (k[:-1], LazyField(v)) if k.endswith('_') else (k, v)


def _process_lazy_args(d):
    if isinstance(d, RecordBase):
        return d.dict_
    if not isinstance(d, T.Mapping):
        d = dict(d)
    return dict(_process_lazy_arg(k, v) for k, v in d.items())


def _check_key(record, key, error_type=KeyError):
    if key not in record.dict_:
        raise error_type(f'{repr(key)} not in available keys: {str(list(record.keys()))[1:-1]}')


class RecordBase(abc.Collection, ABC):  # Sized, Iterable len, iter
    __slots__ = "dict_"

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("All arguments but the first one must be keyword arguments."
                             + " The optional positional argument can only be a Record or Mapping.")
        self.dict_ = _process_lazy_args(kwargs)
        if len(args) == 1:
            self.dict_ = dict(_process_lazy_args(args[0]), **self.dict_)
        if any(isinstance(k, int) for k in self.dict_.keys()):
            raise ValueError("Record keys must be non-ints.")
        if int(os.environ.get('VIDLU_EAGER_RECORD', '0'))==1:
            self.evaluate()
            if 'classes' in self.keys() and self.classes[0] is None:
                breakpoint()

    def __getitem__(self, key):
        _check_key(self, key, KeyError)
        val = self.dict_[key]
        if isinstance(val, LazyField):
            val = self.dict_[key] = val(self)
        return val

    def __setitem__(self, key, value):
        self.dict_.__setitem__(*_process_lazy_arg(key, value))

    def __len__(self):  # Sized
        return len(self.dict_)

    def __getattr__(self, key):  # namespace
        _check_key(self, key, AttributeError)
        return self[key]

    def __eq__(self, other):
        if isinstance(other, RecordBase):
            return self.dict_ == other.dict_
        else:
            return all(a == b for a, b in zip(self.items(), other.items()))

    def __getstate__(self):  # evaluates everything
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        self.dict_ = state

    def __str__(self):
        return self._to_string(str)

    def __repr__(self):
        return self._to_string(repr)

    def evaluate(self):
        for k in self.keys():
            _ = self[k]

    def is_evaluated(self, key):
        return not isinstance(self.dict_[key], LazyField)

    def join(self, other: 'RecordBase', *others, overwrite=False):  # sequence
        if len(others) > 0:
            return reduce(lambda a, b: a.join(b, overwrite=overwrite), [self, other, *others])
        if not overwrite and any(a in self.keys() for a in other.keys()):
            raise ValueError("Sets of field names should be disjoint if `overwrite == False`.")
        return Record(self, **other.dict_)

    def update(self, other):  # Mapping
        self.dict_.update(_process_lazy_args(other))

    def keys(self):  # Mapping
        return self.dict_.keys()

    def values(self):  # Mapping
        return ListRecordView(self)  # TODO

    def items(self):  # Mapping
        return ItemsRecordView(self)  # TODO

    def _to_string(self, func=repr):
        fields = ", ".join(
            [f"{k}={func(self[k])}" if type(self).is_evaluated(self, k) else f"{k}=<unevaluated>"
             for k in self.keys()])
        return f"{type(self).__name__}({fields})"


class Record(RecordBase, abc.Sequence):  # Sized, Iterable len, iter
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

    def __getitem__(self, key: T.Union[int, str, T.Sequence[int], T.Sequence[str]]):
        if isinstance(key, slice):
            return self[list(range(*key.indices(len(self))))]
        elif isinstance(key, abc.Sequence) and not isinstance(key, str):
            if len(key) == 0:
                return Record()
            if isinstance(key[0], int):
                all_keys = tuple(self.keys())
                key = [all_keys[i] for i in key]
            return Record({k: self.dict_[k] for k in key})
        elif isinstance(key, int):
            key = tuple(self.keys())[key]
        return super().__getitem__(key)

    def __iter__(self):  # Sequence (returns values)
        return iter(self.values())

    def __contains__(self, item):  # Mapping
        raise TypeError("`in` operator not supported. Use the `values` or the `keys` method.")


class DictRecord(RecordBase, abc.Mapping):  # Sized, Iterable len, iter
    def __iter__(self):
        return iter(self.dict_.keys())

    def __contains__(self, item):
        return any(v == item for v in self)


def arrange(r, field_names):
    return Record({**r[field_names].dict_, **r.dict_})


class RecordView(Record):
    def __init__(self, record):
        super().__init__()
        self.dict_ = record.dict_


class DictRecordView(RecordView):
    def __iter__(self):
        return iter(self.dict_.keys())

    def __contains__(self, item):
        return any(v == item for v in self)

    def _to_string(self, func=repr):
        fields = ", ".join([k for k in self])
        return f"{type(self).__name__}({fields})"


class ListRecordView(RecordView):
    def __iter__(self):
        return (self[k] for k in self.dict_.keys())

    def __contains__(self, item):
        return any(v == item for v in self)

    def _to_string(self, func=repr):
        fields = ", ".join(
            [f"{func(self[k])}" if type(self).is_evaluated(self, k) else f"<unevaluated>"
             for k in self])
        return f"{type(self).__name__}({fields})"


class ItemsRecordView(RecordView):
    def __iter__(self):
        return ((k, self[k]) for k in self.dict_.keys())

    def __contains__(self, item):
        return any(v == item for v in self)
