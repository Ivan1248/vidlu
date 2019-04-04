import inspect
import select
import sys
import platform
import os
import hashlib
import urllib.request

from tqdm import tqdm


# Inspection #######################################################################################

def class_initializer_locals_c():
    """
    Returns arguments of the initializer of a subclass if there are no variables
    defined before `super().__init__()` calls in the initializer of the object's
    class.
    Based on `magnet.utils.misc.caller_locals`.
    """
    frame = inspect.currentframe().f_back.f_back

    try:
        locals_ = frame.f_locals
        caller = locals_.pop('self')

        while True:
            frame = frame.f_back
            f_class = frame.f_locals.pop('__class__', None)
            locals_curr = frame.f_locals
            if f_class is None or locals_curr.pop('self', None) is not caller:
                break
            locals_ = locals_curr

        locals_.pop('self', None)
        locals_.pop('__class__', None)
        return locals_
    finally:
        del frame  # to avoid cyclical references, TODO


def locals_c(exclusions=('self', '__class__')):
    """ Locals without `self` and `__class__`. """
    frame = inspect.currentframe().f_back
    try:
        locals_ = frame.f_locals
        for x in exclusions:
            locals_.pop(x, None)
        return locals_
    finally:
        del frame  # to avoid cyclical references, TODO


def find_frame_in_call_stack(frame_predicate, start_frame=-1):
    frame = start_frame or inspect.currentframe().f_back.f_back
    while not frame_predicate(frame):
        frame = frame.f_back
        if frame is None:
            return None
    return frame


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
        else:
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


# Dict #############################################################################################

def get_key(dict_, value):
    for k, v in dict_.items():
        if v == value:
            return k
    raise ValueError("the dictionary doesn't contain the value")


# Downloading ######################################################################################


class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


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
