import builtins
import contextlib
import gzip
import hashlib
import importlib.util
import os
import platform
import select
import sys
import time
import typing as T
import warnings
import weakref
import zipfile
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
import copy
import pickle

from tqdm import tqdm
import numpy as np


# Slicing ##########################################################################################

def slice_len(s, sequence_length):
    # stackoverflow.components/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


# Deep attribute access ############################################################################

def deep_getattr(namespace, path: str):
    names = path.split('.')
    obj = namespace[names[0]] if isinstance(namespace, T.Mapping) else getattr(namespace, names[0])
    for name in names[1:]:
        obj = getattr(obj, name)
    return obj


# Argument broadcasting ############################################################################

def broadcast(obj: T.Union[object, T.Sequence], n: int, seq_type=T.Sequence) -> list:
    if isinstance(obj, seq_type):
        if len(obj) == 1:
            return list(obj) * n
        elif len(obj) != n:
            raise RuntimeError(f"`obj` already is a `Sequence` but its size ({len(obj)}) is "
                               f"not `n` = {n}. Check whether `batch_size` and"
                               f" `evaL_batch_size` are correctly set.")
        return obj
    return [obj] * n


# Import module ####################################################################################

def import_module(path):  # from morsic
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Event ############################################################################################

class RemovableHandle(object):
    """An object that can be used to remove a handler from an event.

    It can be removed by calling the `remove` method or by using it as a
    context manager.
    """

    def __init__(self, event, handler):
        self.event = weakref.ref(event)
        self.handler = weakref.ref(handler)

    def remove(self):
        if None not in (event := self.event(), handler := self.handler()):
            event.remove_handler(handler)

    def __call__(self, *args, **kwargs):
        self.handler(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()


class Event:
    """Event implementation.

    Example:
        >>> ev = Event()
        >>> rh = ev.add_handler(print)
        >>> ev("Hello!")
        Hello!
        >>> rh.remove()
        >>> ev("Hello!")
        >>> with ev.add_handler(print) as rh:
        >>>    ev("african")
        african
        >>> ev("swallow")
    """

    def __init__(self):
        self.handlers = []

    def add_handler(self, handler: callable):
        """Adds an event handler (callback) and returns a removable handle.

        Args:
            handler (callable): a function with a corresponding signature.

        Returns:
            RemovableHandle: an object that can be used to remove the handler.
        """
        self.handlers.append(handler)
        return RemovableHandle(self, handler)  # can be used as context manager or decorator

    def handler(self, handler: callable):
        """Adds an event handler (callback) and returns it. It should only be
        used as a decorator. `add_handler` should be used otherwise."""
        self.handlers.append(handler)
        return handler

    def remove_handler(self, handler: callable):
        self.handlers.remove(handler)

    def __call__(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)


# Console ##########################################################################################

def try_input(default=None):
    """Returns `None` if the standard input is empty. Otherwise it returns a
    line like `input` does.
    """

    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        if sys.stdin.isatty():
            return select.select([sys.stdin], [], [], 0)[0]

    try:
        return input() if input_available() else default
    except UnicodeDecodeError as e:
        warnings.warn(e)
        return default


def query_user(question, default=None, timeout=np.inf, options=None):
    options = options or dict(y=True, n=False)
    if timeout is not np.inf and default not in options:
        raise ValueError(f"`default` should have a value from {set(options.keys())} when `timeout`"
                         + " is finite.")

    options_str = "/".join(f"{{{c}}}" if c == default else c for c in options)
    while True:
        sys.stdout.write(f'{question} [{options_str}]: ')
        sys.stdout.flush()
        sw = Stopwatch().start()
        inp = no_input = id(sw)
        while sw.time < timeout:
            time.sleep(0.1)
            if (inp := try_input(default=no_input)) is not no_input:
                print()
                break
        if inp in [no_input, ""]:
            return options[default]
        elif inp in options:
            return options[inp]
        else:
            print(f"Please respond with either of {', '.join(options.keys())}.")


# Archive files ####################################################################################


def extract_zip(path, dest_path):
    with zipfile.ZipFile(path) as archive:
        files = list(archive.namelist())
        for filename in tqdm(files, f"Extracting {path} to {dest_path}"):
            archive.extract(filename, dest_path)
    return files


def extract_gz(path, dest_path):
    with gzip.open(path, 'rb') as gz:
        with open(dest_path, 'wb') as raw:
            raw.write(gz.read())
    return dest_path


# Downloading ######################################################################################

class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_if_not_downloaded(url, output_path):
    if not Path(output_path).exists():
        download(url, output_path, md5=None)
        return True
    return False


def download(url, output_path, md5=None):
    import urllib.request

    def check_integrity(fpath, md5):
        # from torchvision.datasets.utils
        if not os.path.isfile(fpath):
            return False
        md5o = hashlib.md5()
        with open(fpath, 'rb') as f:
            # read in 1MB chunks
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5o.update(chunk)
        md5c = md5o.hexdigest()
        return md5c == md5

    if md5 is not None and os.path.isfile(output_path) and check_integrity(output_path, md5):
        print(f'Using downloaded and verified file: {output_path}')
    with _DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc="Downloading " + url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# Mappings #########################################################################################

def fuse(*dicts, overriding=None, ignore_if_equal=True, factory=None):
    factory = factory or type(dicts[0])

    if overriding is None and ignore_if_equal is False:
        return factory(*dicts)

    result = factory(**dicts[0])
    for d in dicts[1:]:
        for k, v in d.items():
            if k in result and not (ignore_if_equal and result[k] is v):
                raise RuntimeError(f"Key '{k}' is already assigned.")
            result[k] = v

    overriding = overriding or factory()
    result.update(overriding)

    return result


def dict_difference(a, b):
    return type(a)({k: v for k, v in a.items() if k not in b})


def update_existing_items(dest, src, copy=False):
    if copy:
        dest = dest.copy()
    dest.update({k: v for k, v in src.items() if k in dest})
    return dest


# Context managet timer ############################################################################

class Stopwatch:
    """A stopwatch that can be used as a context manager.

    Example:
        with Stopwatch as sw:
            sleep(1)
            assert sw.running and sw.time >= 1.
        assert not sw.running and sw.time >= 1.

    Example:
        sw = Stopwatch().start()
        sleep(1)
        assert sw.running and sw.time >= 1.
        sw.stop()
        assert not sw.running and sw.time >= 1.

        sw.reset()
        assert not sw.running and sw.time == sw.start_time == 0
    """
    __slots__ = '_time_func', 'start_time', '_time', 'running'

    def __init__(self, time_func=time.time):
        self._time_func = time_func
        self.reset()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return f"Stopwatch(time={self.time})"

    @property
    def time(self):
        return self._time + self._time_func() - self.start_time if self.running else self._time

    def reset(self):
        self._time = 0.
        self.start_time = None
        self.running = False
        return self

    def start(self):
        if self.running:
            self.reset()
        self.start_time = self._time_func()
        self.running = True
        return self

    def stop(self):
        if self.running:
            self._time = self.time
            self.running = False
        return self._time


# Shared arrays ####################################################################################

def to_shared_array(x):
    x_shared = RawArray(np.ctypeslib.as_ctypes_type(x.dtype), x.size)
    x_shared = np.frombuffer(x_shared, dtype=x.dtype).reshape(x.shape)
    x_shared[:] = x
    return x_shared


# Progresss bar ####################################################################################

def item_pbar(seq, transform=str):
    tseq = tqdm(seq)
    for x in tseq:
        tseq.set_description(transform(x))
        tseq.update()
        yield x


def key_pbar(seq):
    return item_pbar(seq, lambda x: str(x[0]))


# Indent print #####################################################################################


@contextlib.contextmanager
def indent_print(*args, indent="   "):
    if len(args) > 0:
        print(*args)
    orig_print = builtins.print

    def ind_print(*args, **kwargs):
        orig_print(indent[:-1], *args, **kwargs)

    builtins.print = ind_print
    yield
    builtins.print = orig_print


# Checks ###########################################################################################

def check_arg_type(name, value, type_):
    if not isinstance(value, type_):
        raise TypeError(f"The type of the argument {name} is {type(value).__qualname__}, but should"
                        f"be {getattr(type_, '__qualname__', type_)}.")


def check_value_in(name, value, values: T.Collection):
    if value not in values:
        raise TypeError(f"{name} should have a value from {values}, not {value}.")


# Typing ###########################################################################################


class TypeOperationBase:
    @classmethod
    def __instancecheck__(cls, obj):
        print(type(cls))
        return cls.__subclasscheck__(cls, type(obj))


class SubclassCheck(TypeOperationBase):
    def __init__(self, check: T.Callable[[type], bool]):
        self.__class__ = copy(self.__class__)
        self.__class__.check = check

    @classmethod
    def __subclasscheck__(cls, subclass):
        return cls.check(subclass)


class TypeOperation(TypeOperationBase):
    classes = None

    def __init__(self, *classes):
        self.__class__ = copy(self.__class__)
        self.__class__.classes = classes

    @classmethod
    def __subclasscheck__(cls, subclass):
        print(type(cls))
        return issubclass(subclass, cls.classes)

    @classmethod
    def __instancecheck__(cls, obj):
        print(type(cls))
        return cls.__subclasscheck__(cls, type(obj))


class Union(TypeOperationBase):
    @classmethod
    def __subclasscheck__(cls, subclass):
        print(type(cls))
        return issubclass(subclass, cls.classes)


class Intersection(TypeOperationBase):
    @classmethod
    def __subclasscheck__(cls, subclass):
        return all(issubclass(subclass, c) for c in cls.classes)


# Serialization ####################################################################################

def pickle_sizeof(obj):
    """An alternative to `sys.getsizeof` which works for lazily initialized objects (e.g. objects of
    type `vidlu.data.Record`) that can be much larger when pickled.

    Args:
        obj: the object to be pickled.

    Returns:
        int: the size of the pickled object in bytes.
    """
    return len(pickle.dumps(obj))
