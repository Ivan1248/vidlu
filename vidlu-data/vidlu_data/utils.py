"""Utility functions extracted from vidlu.utils for standalone use."""
import functools
import os
import pickle
import platform
import re
import select
import sys
import tempfile
import time
import warnings
from inspect import signature
from pathlib import Path
import datetime as dt

import numpy as np
import torch
import blosc
import PIL.Image as pimg


# from vidlu.utils.func

def positional_param_count(func) -> int:
    if not callable(func):
        raise ValueError("The argument should be a function.")
    return sum(1 for param in signature(func).parameters.values()
               if param.kind == param.POSITIONAL_OR_KEYWORD)


# from vidlu.utils.misc

def slice_len(s, sequence_length):
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def pickle_sizeof(obj):
    return len(pickle.dumps(obj))


def _try_input(default=None):
    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        return sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]

    try:
        return input() if input_available() else default
    except UnicodeDecodeError as e:
        warnings.warn(e)
        return default


class Stopwatch:
    """A stopwatch that can be used as a context manager."""

    __slots__ = '_time_func', 'start_time', '_prev_time', 'running'

    def __init__(self, time_func=time.time):
        self._time_func = time_func
        self.reset()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return f"Stopwatch(time={self.time}, running={self.running})"

    @property
    def time(self):
        if self.running:
            return self._prev_time + self._time_func() - self.start_time
        else:
            return self._prev_time

    def reset(self):
        self._prev_time = 0.
        self.start_time = None
        self.running = False
        return self

    def start(self):
        if self.running:
            warnings.warn("Stopwatch already running.")
        else:
            self.start_time = self._time_func()
            self.running = True
        return self

    def stop(self):
        if self.running:
            self._prev_time = self.time
            self.running = False
        else:
            warnings.warn("Stopwatch is already not running.")
        return self._prev_time


def query_user(question, default=None, timeout=np.inf, options=None):
    options = options or dict(y=True, n=False)
    if timeout is not np.inf and default not in options:
        raise ValueError(f"`default` should have a value from {set(options.keys())} when `timeout`"
                         + " is finite.")
    options_str = "/".join(f"{{{c}}}" if c == default else c for c in options)
    while True:
        sys.stdout.write(f'{question} ' + (
            "" if timeout is None else f'(timeout {timeout}s)') + f' [{options_str}]: ')
        sys.stdout.flush()
        sw = Stopwatch().start()
        inp = no_input = id(sw)
        while sw.time < timeout:
            if (inp := _try_input(default=no_input)) is not no_input:
                print()
                break
            time.sleep(0.1)
        if inp in [no_input, ""]:
            return options[default]
        elif inp in options:
            return options[inp]
        else:
            print(f"Please respond with either of {', '.join(options.keys())}.")


# from vidlu.utils.path

def find_in_ancestors(start, subpath, include_start=False, ignore_broken_symlinks=False):
    start = Path(start).absolute()
    if include_start:
        start /= "_"
    for anc in start.parents:
        candidate = anc / subpath
        if candidate.exists(follow_symlinks=not ignore_broken_symlinks):
            return candidate
    raise FileNotFoundError(f"No ancestor of {start} has a child {subpath}.")


def _split_long_name(name, max_length=255):
    result, remainder = [], name
    while len(remainder) > max_length:
        result.append(remainder[:max_length])
        remainder = remainder[max_length:]
    result.append(remainder)
    return result


def _split_long_names(path: Path, max_length=255):
    partses = [_split_long_name(p, max_length) for p in path.parts]
    return functools.reduce(list.__add__, partses, [])


def to_valid_path(path, split_long_names=False, max_name_length=255):
    path = str(path).strip()
    allowed = r"-\w.,\'\\/!#$%^&()_+=@{}\[\]"
    path = Path(re.sub(f"(?u)[^{allowed}]", "+", str(path)))
    if split_long_names:
        parts = _split_long_names(path, max_name_length)
        path = Path(os.path.join(*parts))
    return path


def time_since_access(file):
    access_time = dt.datetime.utcfromtimestamp(Path(file).stat().st_atime)
    return dt.datetime.utcnow() - access_time


def create_file_atomic(path, write_action, mode="w+b"):
    """Writes to a temp file then atomically renames it to `path`."""
    tmp = tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=Path(path).parent)
    try:
        write_action(tmp.file)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    else:
        tmp.close()
        os.rename(tmp.name, path)


# from vidlu.utils.storage.compressors

class CompressedObject:
    def __init__(self, compressor, data):
        self.data = compressor.compress(data)
        self._decompress = compressor.decompress

    def decompress(self):
        return self._decompress(self.data)


class Compressor:
    def compress(self, obj):
        raise NotImplementedError

    def decompress(self, obj):
        raise NotImplementedError


class NonCompressor:
    def compress(self, obj):
        return obj

    def decompress(self, obj):
        return obj


class NumpyCompressor(Compressor):
    def compress(self, arr):
        c = blosc.compress_ptr(arr.__array_interface__['data'][0], arr.size, arr.dtype.itemsize,
                               clevel=9, cname='lz4hc', shuffle=blosc.SHUFFLE)
        return c, arr.shape, arr.dtype

    def decompress(self, obj):
        c, shape, dtype = obj
        arr = np.empty(shape, dtype)
        blosc.decompress_ptr(c, arr.__array_interface__['data'][0])
        return arr


class TorchCompressor(Compressor):
    def __init__(self, numpy_compressor_f=NumpyCompressor):
        self.numpy_compressor = numpy_compressor_f()

    def compress(self, obj):
        return self.numpy_compressor.compress(obj.numpy())

    def decompress(self, obj):
        return torch.from_numpy(obj)


class PILCompressor(Compressor):
    def __init__(self, numpy_compressor_f=NumpyCompressor):
        self.numpy_compressor = numpy_compressor_f()

    def compress(self, obj):
        mode = obj.mode
        arr = np.array(obj)
        carr = self.numpy_compressor.compress(arr)
        return carr, mode

    def decompress(self, obj):
        carr, mode = obj
        arr = self.numpy_compressor.decompress(carr)
        return pimg.fromarray(arr, mode=mode)


class DefaultCompressor(Compressor):
    def compress(self, obj):
        compressor = (NumpyCompressor() if isinstance(obj, np.ndarray) else
                      PILCompressor() if isinstance(obj, pimg.Image) else
                      NonCompressor())
        return CompressedObject(compressor, obj)

    def decompress(self, obj):
        return obj.decompress()
