import warnings
from pathlib import Path
import os
from argparse import Namespace

from vidlu.utils import path

__all__ = ("DATASETS", "EXPERIMENTS", "CACHE", "SAVED_STATES", "PRETRAINED")


def _find(path_end):
    try:
        return Path(path.find_in_ancestors(__file__, path_end))
    except FileNotFoundError:
        warnings.warn(f'Cannot find directory "{path_end}" in ancestors of {__file__}.')
        return None


def filter_valid(items):
    return [x for x in items if x is not None]


# Datasets
DATASETS = filter_valid([Path(p) if (p := os.environ.get("VIDLU_DATASETS", None)) else None,
                         _find('data/datasets'), _find('datasets'), Path('/tmp/')])

# Pre-trained parameters
PRETRAINED = Path(os.environ.get("VIDLU_PRETRAINED", None) or _find('data/pretrained_parameters'))

# Cache and experimental results/states
EXPERIMENTS = Path(os.environ.get("VIDLU_EXPERIMENTS", None) or _find('data/experiments'))

# Cache
CACHE = EXPERIMENTS / 'cache'

# Experimental results/states
SAVED_STATES = EXPERIMENTS / 'states'

CACHE.mkdir(exist_ok=True)
SAVED_STATES.mkdir(exist_ok=True)

for k in __all__:
    path = globals()[k] if k != "DATASETS" else DATASETS[0]
    if not path.exists():
        raise FileNotFoundError(f'Directory {k}="{path}" does not exist.')
    if not path.is_dir():
        raise FileNotFoundError(f'{k}="{path}" is not a directory path.')
