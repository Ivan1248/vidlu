from contextlib import contextmanager
import select
import sys
import platform
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
import time
import contextlib
from multiprocessing.sharedctypes import RawArray

from tqdm import tqdm
import urllib.request
import numpy as np


# Slicing ##########################################################################################

def slice_len(s, sequence_length):
    # stackoverflow.components/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


# Event ############################################################################################

class Event:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)
        return handler  # for usage as decorator

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    @contextlib.contextmanager
    def temporary_handler(self, handler):
        self.add_handler(handler)
        yield handler
        self.remove_handler(handler)

    def __call__(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)


# Console ##########################################################################################

def try_input(default=None):
    """
    Returns `None` if the standard input is empty. Otherwise it returns a line
    like `input` does.
    """

    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        if sys.stdin.isatty():
            return select.select([sys.stdin], [], [], 0)[0]

    return input() if input_available() else default


def query_yes_no(question):
    valid = dict(yes=True, y=True, no=False, n=False)
    while True:
        sys.stdout.write(f'{question} [y/n]: ')
        choice = valid.get(input().lower(), None)
        if choice is not None:
            return choice
        print("Please respond with either 'yes', 'no', 'y', or 'n').")


# Downloading ######################################################################################

class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_if_not_downloaded(url, output_path):
    if not Path(output_path).exists():
        download(url, output_path, md5=None)


def download(url, output_path, md5=None):
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
        if md5c != md5:
            return False
        return True

    if md5 is not None and os.path.isfile(output_path) and check_integrity(output_path, md5):
        print(f'Using downloaded and verified file: {output_path}')
    with _DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc="Downloading " + url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# Iterator functions

def skip(iterator, n=None):
    """Advances the iterator `n` steps. If n is None, the whole iterator is consumed."""
    if n is not None or not (isinstance(n, int) and n > 0):
        raise ValueError(f"`n` should be `None` or a positive integer, not {n}.")

    if n is None:
        for _ in iterator:
            pass
    else:
        for i, _ in enumerate(iterator):
            if i >= n:
                break
    return iterator


def consume(iterator):
    for _ in iterator:
        pass
    return iterator


# Dataclasses

@dataclass
class FieldCheckingClass:
    def __post_init__(self):
        self.check_all_initialized()

    def check_all_initialized(self, invalid_predicate):
        for k, v in self.__dict__:
            if invalid_predicate(v):
                raise TypeError(
                    f"Attribute {k} has invalid value {v} (for an object of type {type(self)}).")


def check_all_initialized(obj, invalid_predicate):
    for k, v in obj.__dict__.items():
        if invalid_predicate(v):
            raise TypeError(
                f"{type(obj).__name__} attribute '{k}' is missing or has invalid value {v} .")


# Mappings


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


# module importer

class Meta(type):
    def __getattr__(cls, key):
        return __import__(key)


class Importer:
    __metaclass__ = Meta


# context manager timer

class CMTimer:
    """Context manager timer"""
    __slots__ = '_time_func', 'start', 'time'

    def __init__(self, time_func=time.time):
        self._time_func = time_func

    def __enter__(self):
        self.start = self._time_func()
        return self

    def __exit__(self, *args):
        self.time = self._time_func() - self.start


# shared array

def to_shared_array(x):
    x_shared = RawArray(np.ctypeslib.as_ctypes_type(x.dtype), x.size)
    x_shared = np.frombuffer(x_shared, dtype=x.dtype).reshape(x.shape)
    x_shared[:] = x
    return x_shared
