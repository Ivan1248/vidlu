from collections.abc import Mapping


class NameDict(Mapping):
    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(
                f"{type(self).__name__} expected at most 1 positional argument, got {len(args)}.")
        for k, v in dict(*args, **kwargs).items():
            setattr(self, k, v)

    def __repr__(self):
        arg_strings = [f'{name}={value}' for name, value in self._get_kwargs()]
        return f"{type(self).__name__}({', '.join(arg_strings)})"

    def __eq__(self, other):
        if not isinstance(other, NameDict):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        if not name.isidentifier():
            raise KeyError('A NameDict can only have keys that are valid identifiers.')
        setattr(self, name, value)

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
        self.__dict__.update(*args, **kwargs)


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
