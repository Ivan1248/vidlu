import warnings
from pathlib import Path

from vidlu.utils import path


def _find(path_end):
    try:
        return Path(path.find_in_ancestors(__file__, path_end))
    except FileNotFoundError:
        warnings.warn(f'Cannot find directory "{path_end}".')
        return None


DATASETS = [Path('/tmp/'), _find('data/datasets'), _find('datasets')]
DATASETS = [p for p in DATASETS if p]
EXPERIMENTS = _find('data/experiments')
CACHE = EXPERIMENTS / 'cache'
SAVED_STATES = EXPERIMENTS / 'states'
PRETRAINED = _find('data/pretrained_parameters')

CACHE.mkdir(exist_ok=True)
SAVED_STATES.mkdir(exist_ok=True)
