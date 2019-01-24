from collections.abc import Mapping


class NamespaceDict(Mapping):
    def __init__(self, *args, **kwargs):
        assert len(args) <= 1
        if len(args) == 1:
            kwargs = {**args[0], **kwargs}
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __repr__(self):
        arg_strings = [f'{name}={value}' for name, value in self._get_kwargs()]
        return f"{type(self).__name__}({', '.join(arg_strings)})"

    def __eq__(self, other):
        if not isinstance(other, NamespaceDict):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        assert name.isidentifier()
        setattr(self, name, value)

    def __getattr__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def _get_kwargs(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def as_dict(self):
        return self.__dict__


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
