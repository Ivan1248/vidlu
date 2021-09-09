import warnings
from pathlib import Path
import os
from functools import cached_property

from vidlu.utils.path import find_in_ancestors


def _find(path_end, start=__file__, warn=False):
    try:
        return Path(find_in_ancestors(start, path_end))
    except FileNotFoundError:
        if warn:
            warnings.warn(f'Cannot find directory "{path_end}" in ancestors of {__file__}.')
        return None


def opt(func, x):
    return func(x) if x else None


def _check_dir_path(path, kind=None):
    if path is None:
        raise FileNotFoundError(f'No {kind} directory provided or found.')
    if not path.exists():
        raise FileNotFoundError(f'Directory "{path}" does not exist.')
    if not path.is_dir():
        raise FileNotFoundError(f'"{path}" is not a directory.')


class _Dirs:
    @cached_property
    def data(self):
        """Optional root directory that can contain "datasets", "pretrained"
        and "experiments" directories.
        """
        return opt(Path, os.environ.get("VIDLU_DATA", None))

    @cached_property
    def datasets(self):
        """List of directories that contain datasets"""
        datasets = [self._get_path("datasets"), _find('datasets'), _find('datasets', start='tmp/_')]
        datasets = [x for x in datasets if x is not None]
        _check_dir_path(None if len(datasets) == 0 else datasets[0], "datasets")
        return datasets

    @cached_property
    def pretrained(self):
        """Pretrained parameters"""
        pretrained = self._get_path("pretrained")
        _check_dir_path(pretrained, "pretrained")
        return pretrained

    @cached_property
    def experiments(self):
        """Cache and experimental results/states"""
        experiments = self._get_path("experiments")
        _check_dir_path(experiments, "experiments")
        return experiments

    @cached_property
    def cache(self):
        """Various cache"""
        cache = self._get_path("cache")
        _check_dir_path(cache, "cache")
        return cache

    @cached_property
    def saved_states(self):
        """Experimental results or incomplete training states"""
        saved_states = self.experiments / 'states'
        saved_states.mkdir(exist_ok=True)
        return saved_states

    def _get_path(self, dir_name: str):
        return opt(Path, self.data / dir_name if self.data else (
                os.environ.get(f"VIDLU_{dir_name.upper()}", None)
                or _find(f"data/{dir_name}", warn=True)))


_dirs = _Dirs()


def __getattr__(name):
    return getattr(_dirs, name)


def __dir__():
    return "data", "datasets", "pretrained", "experiments", "cache", "saved_states"
