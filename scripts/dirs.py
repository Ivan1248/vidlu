import warnings
from pathlib import Path
import os

from vidlu.utils import path

__all__ = ("datasets", "experiments", "cache", "saved_states", "pretrained")


def _find(path_end, start=__file__):
    try:
        return Path(path.find_in_ancestors(start, path_end))
    except FileNotFoundError:
        warnings.warn(f'Cannot find directory "{path_end}" in ancestors of {__file__}.')
        return None


def opt_path(x):
    return Path(x) if x else None


def filter_valid(items):
    return [x for x in items if x is not None]


DATA = opt_path(os.environ.get("VIDLU_DATA", None))

# Datasets
datasets = filter_valid(
    [opt_path(DATA / "datasets" if DATA else os.environ.get("VIDLU_DATASETS", None)),
     _find('data/datasets'), _find('datasets'), _find('datasets', start='tmp/_')])
if len(datasets) ==0:
    raise FileNotFoundError('No directory containing datasets provided or found.')

# Pre-trained parameters
pretrained = opt_path(DATA / "pretrained" if DATA else
                      os.environ.get("VIDLU_PRETRAINED", None) or _find('data/pretrained'))
if pretrained is None:
    raise FileNotFoundError('"pretrained" directory not provided or found.')

# Cache and experimental results/states
experiments = opt_path(DATA / "experiments" if DATA else
                       os.environ.get("VIDLU_EXPERIMENTS", None) or _find('data/experiments'))
if experiments is None:
    raise FileNotFoundError('"experiments" directory not provided or found.')

# Cache
cache = experiments / 'cache'

# Experimental results/states
saved_states = experiments / 'states'

cache.mkdir(exist_ok=True)
saved_states.mkdir(exist_ok=True)

for k in __all__:
    path = globals()[k] if k != "datasets" else datasets[0]
    if not path.exists():
        raise FileNotFoundError(f'Directory {k}="{path}" does not exist.')
    if not path.is_dir():
        raise FileNotFoundError(f'{k}="{path}" is not a directory path.')
