import inspect
import select
import sys
import platform
import os
import hashlib
import urllib.request

from tqdm import tqdm


# Inspection #######################################################################################

def clean_locals_from_first_initializer():
    """
    Returns arguments of the constructor of a subclass if `super().__init__()`
    is the first statement in the subclass' `__init__`. Taken from MagNet
    (https://github.components/MagNet-DL/magnet/blob/master/magnet/utils/misc.py) and
    modified. Originally `magnet.utils.misc.caller_locals`.
    """
    frame = inspect.currentframe().f_back.f_back

    try:
        l = frame.f_locals

        f_class = l.pop('__class__', None)
        caller = l.pop('self')
        while f_class is not None and isinstance(caller, f_class):
            l.pop('args', None)
            args = frame.f_locals.pop('args', None)
            l.update(frame.f_locals)
            if args is not None:
                l['args'] = args

            l.pop('self', None)
            frame = frame.f_back
            f_class = frame.f_locals.pop('__class__', None)

        l.pop('self', None)
        l.pop('__class__', None)
        return l
    finally:
        del frame


def clean_locals():
    locals_ = inspect.currentframe().f_back.f_locals
    locals_.pop('self', None)
    locals_.pop('__class__', None)
    return locals_


def find_frame_in_call_stack(frame_predicate, start_frame=None):
    frame = start_frame or inspect.currentframe().f_back.f_back
    try:
        while not frame_predicate(frame):
            frame = frame.f_back
        return frame
    finally:
        del frame


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

    def handler(self):
        def decorator(f):
            self.add_handler(f)
            return f

        return decorator

    def __call__(self, *args, **kwargs):
        for h in self.handlers:
            h(*args, **kwargs)


# Console ##########################################################################################

def try_input():
    """
    Returns `None` if the standard input is empty. Otherwise it returns a line
    like `input` does.
    """

    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0)[0]

    return input() if input_available() else None


# Dict #############################################################################################

def get_key(dict_, value):
    for k, v in dict_.items():
        if v is value:
            return k
    raise ValueError("the dictionary doens't ontain the value")


# Downloading ######################################################################################

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


class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path, md5=None):
    if md5 is not None and os.path.isfile(output_path) and check_integrity(output_path, md5):
        print(f'Using downloaded and verified file: {output_path}')
    with _DownloadProgressBar(unit='B', unit_scale=True,
                              miniters=1, desc="Downloading " + url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
