from vidlu.utils import path


def _find(path_end):
    try:
        return path.find_in_ancestor(__file__, path_end)
    except FileNotFoundError:
        print(f"dirs.py: WARNING: Cannot find {path_end}.")
        return None


DATASETS = _find('data/datasets')
CACHE = _find('data/cache')

SAVED_STATES = _find('data/states')
PRETRAINED = _find('data/pretrained_parameters')
LOGS = _find('data/logs')
