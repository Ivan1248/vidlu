import collections
import pickle
import os
import collections.abc as abc
import typing as T
from pathlib import Path
import warnings
import shutil


def _augment_key_error_message(dct, e, error_type=None):
    if error_type is None:
        error_type = type(e)
    return error_type(f"{e.args[0]}. Available keys: {list(dct.keys())}", *e.args[1:])


class NameDict(abc.MutableMapping):
    """

    Keys equal to method names "get", "keys", "values", "items", "pop", "popitem", "clear",
    "update", and "setdefault" cannot be used with the dot (attribute access) operator.

    The `vars` function can be used to get the `dict` representation of a `NameDict` object.
    >>> assert vars(nd) is nd.__dict__
    >>> assert NameDict(vars(nd)) == nd

    Implementations of such a type that inherit `dict` and define less methods can cause problems
    with multiple inheritance. Example:
    >>> class AttrDict(dict):
    >>>     def __init__(self, *args, **kwargs):
    >>>        dict.__init__(self, *args, **kwargs)
    >>>        self.__dict__ = self
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 1:
            raise TypeError(f"{type(self).__name__} got more than 1 positional argument.")
        self.update(*args, **kwargs)

    def __repr__(self):
        arg_strings = [f'{name}={repr(value)}' for name, value in self.items()]
        return f"{type(self).__name__}({', '.join(arg_strings)})"

    def __eq__(self, other):
        return vars(self) == vars(other) if type(other) == type(self) else NotImplemented

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, name):
        try:
            return self.__dict__[name]
        except KeyError as e:
            raise _augment_key_error_message(self, e)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __getattr__(self, key):
        """This is defined just for better error messages."""
        try:
            return self.__dict__[key]
        except KeyError as e:
            raise _augment_key_error_message(self, e, AttributeError)  # Must be AttributeError

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def update(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


class Registry(collections.UserDict):
    def __init__(self, filter_=None):
        super().__init__()
        self.filter = (lambda x: x) if filter_ is None else filter_

    def register(self, *args):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if len(args) == 1:
            obj = args[0]
            name = obj.__name__
        else:
            name, obj = args
        self[name] = obj
        return obj  # in case that it is used as a decorator

    def register_from(self, namespace):
        for k, v in vars(namespace).items():
            if self.filter(v):
                self.register(k, v)


class SingleWriteDict(dict):
    def __init__(self, *a, **k):
        dict.__init__(self, *a, **k)

    def __setitem__(self, key, value):
        assert key not in self, f"Key override not allowed (key: {key})"
        dict.__setitem__(self, key, value)

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, T.Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(copy.deepcopy(self.items()))

    def __repr__(self):
        return f'SingleWriteDict({dict.__repr__(self)})'


class FileDict(abc.MutableMapping):
    __slots__ = ("path", "load_proc", "save_proc", "_dict")

    def __init__(self, path: os.PathLike, load_proc=pickle.load, save_proc=pickle.dump,
                 load_in_init=False,
                 corrupt_file_action: T.Union[T.Literal['delete', 'error']] = 'error'):
        self.path = Path(path)
        self.load_proc, self.save_proc = load_proc, save_proc
        self._dict = dict()
        if load_in_init and self.path.exists():
            try:
                self.load()
            except (EOFError, RuntimeError) as e:
                message = f"Error loading FileDict from file {self.path}: {e}"
                if corrupt_file_action == 'error':
                    raise EOFError(message) from e
                else:
                    self.path.unlink()
                    warnings.warn(message)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self._dict)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, name):
        return self._dict[name]

    def __setitem__(self, name, value):
        self._dict[name] = value
        self.save()

    def __delitem__(self, name):
        del self._dict[name]
        self.save()

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __getstate__(self):
        return self._dict

    def __setstate__(self, state):
        self._dict = state
        self.save()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def clear(self):
        self._dict.clear()

    def pop(self, *args):
        result = self._dict.pop(*args)
        self.save()
        return result

    def load(self):
        with open(self.path, "rb") as file:
            self._dict.clear()
            self._dict.update(self.load_proc(file))

    def save(self):
        with open(self.path, "wb") as file:
            self.save_proc(self._dict, file)


class FSDict(collections.UserDict):
    def __init__(self, path=None):
        super().__init__()
        from .storage.compressors import DefaultCompressor
        self.compressor = DefaultCompressor()
        self.path = path
        os.makedirs(path, exist_ok=True)

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def _get_path(self, key):
        return self.path / key

    def _path_to_key(self, path):
        return path.name

    def _save(self, cache_path, obj):
        with open(cache_path, 'wb') as cache_file:
            cobj = self.compressor.compress(obj)
            pickle.dump(cobj, cache_file, protocol=4)

    def _load(self, cache_path):
        with open(cache_path, 'rb') as cache_file:
            cobj = pickle.load(cache_file)
            return self.compressor.decompress(cobj)

    def __getitem__(self, key):
        return self._load(self._get_path(key))

    def __setitem__(self, key, value):
        self._save(self._get_path(key), value)

    def __delitem__(self, key):
        self._get_path(key).unlink()

    def update(self, d):
        for k, v in d.items():
            self[k] = v

    def keys(self):
        return list(map(self._path_to_key, self.path.iterdir()))

    def __contains__(self, key):
        return self._get_path(key).exists()

    def __len__(self):
        return len(self.keys())
