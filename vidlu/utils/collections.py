import pickle
import os
from collections.abc import MutableMapping, Mapping
import typing as T
from pathlib import Path
import warnings


class NameDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 1:
            raise TypeError(
                f"{type(self).__name__} expected at most 1 positional argument, got {len(args)}.")
        self.update(*args, **kwargs)

    def __repr__(self):
        arg_strings = [f'{name}={value}' for name, value in self._get_kwargs()]
        return f"{type(self).__name__}({', '.join(arg_strings)})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        del self.__dict__[name]

    def __getattr__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def _get_kwargs(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def as_dict(self):
        return self.__dict__

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            setattr(self, k, v)


class SingleWriteDict(dict):
    def __init__(self, *a, **k):
        dict.__init__(self, *a, **k)

    def __setitem__(self, key, value):
        assert key not in self, f"Key override not allowed (key: {key})"
        dict.__setitem__(self, key, value)

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
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


class FileDict(MutableMapping):
    __slots__ = ("path", "load_proc", "store_proc", "_dict")

    def __init__(self, path: os.PathLike, load_proc=pickle.load, save_proc=pickle.dump,
                 error_on_corrupt_file=False):
        self.path = Path(path)
        self.load_proc, self.store_proc = load_proc, save_proc
        self._dict = dict()
        if self.path.exists():
            try:
                self.load()
            except EOFError as ex:
                message = f"Error loading FileDict from file {self.path}: {ex}"
                if error_on_corrupt_file:
                    raise EOFError(message)
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

    def _get_kwargs(self):
        return self._dict.items()

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
            self._dict.update(pickle.load(file))

    def save(self):
        with open(self.path, "wb") as file:
            pickle.dump(self._dict, file)
