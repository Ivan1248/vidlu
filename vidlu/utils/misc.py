import select
import sys
import platform
import os
import hashlib
import urllib.request

from tqdm import tqdm


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
