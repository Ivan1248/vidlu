import select
import sys
import platform
import os
import hashlib
from pathlib import Path
import time
import contextlib
from multiprocessing.sharedctypes import RawArray
import urllib.request

from tqdm import tqdm
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
        return md5c == md5

    if md5 is not None and os.path.isfile(output_path) and check_integrity(output_path, md5):
        print(f'Using downloaded and verified file: {output_path}')
    with _DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc="Downloading " + url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


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


# context manager timer

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
        return f"BlockTimer(time={self.time})"

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
            raise RuntimeError("Stopwatch is already running.")
        self.start_time = self._time_func()
        self.running = True
        return self

    def stop(self):
        if self.running:
            self._time = self.time
            self.running = False
        return self._time


# shared array

def to_shared_array(x):
    x_shared = RawArray(np.ctypeslib.as_ctypes_type(x.dtype), x.size)
    x_shared = np.frombuffer(x_shared, dtype=x.dtype).reshape(x.shape)
    x_shared[:] = x
    return x_shared


# progress bar

def item_pbar(seq, transform=str):
    tseq = tqdm(seq)
    for x in tseq:
        tseq.set_description(transform(x))
        tseq.update()
        yield x


def key_pbar(seq):
    return item_pbar(seq, lambda x: str(x[0]))
