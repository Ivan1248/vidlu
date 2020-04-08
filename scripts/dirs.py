import warnings
from pathlib import Path

from vidlu.utils import path


def _find(path_end):
    try:
        return Path(path.find_in_ancestors(__file__, path_end))
    except FileNotFoundError:
        warnings.warn(f'Cannot find directory "{path_end}".')
        return None


DATASETS = [Path('/tmp/'), _find('data/datasets')]
CACHE = _find('data/cache')

SAVED_STATES = _find('data/states')
PRETRAINED = _find('data/pretrained_parameters')
