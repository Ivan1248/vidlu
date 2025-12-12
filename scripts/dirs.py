import warnings
from pathlib import Path
import os
from functools import cached_property

from vidlu.utils.path import find_in_ancestors

__doc__ = """
`scripts/dirs.py` searches for and stores directory paths for datasets, cache, results and other data in the following variables:
-   `datasets: list[Path]` can point to multiple directories, each of which can contain dataset directories.
-   `cache: Path` points to a directory for caching data.
-   `experiments: Path` points to a directory for experiment results. The directory `saved_states = experiments / "states"` is automatically created for storing intermediate and complete training states.
-   `pretrained: Path` points to a directory for pre-trained parameters.

It might be easiest to create the following directory structure. Symbolic links can be useful.

```sh
<ancestor>
├─ .../vidlu/scripts/dirs.py
└─ data
   ├─ cache
   ├─ datasets
   ├─ experiments  # subdirectories created automatically
   │  └─ states
   └─ pretrained
```

The "data" directory can be created in the user home directory by running

```sh
mkdir ~/data ~/data/datasets ~/data/cache ~/data/experiments ~/data/pretrained
```

The "data" directory is found automatically if it is in an ancestor directory of `dirs.py`. Otherwise, the environment variable `VIDLU_DATA` should point to the "data" directory.

The "cache" directory should preferably be on an SSD. "datasets" and other directories, can be on a slower disk. Data from "datasets" is not accessed after being cached.

Alternatively, the paths can be defined individually through multiple environment variables: `VIDLU_DATASETS`, `VIDLU_CACHE`, `VIDLU_PRETRAINED`, and `VIDLU_EXPERIMENTS`.
"""


def _find(path_end, start=__file__, warn=False):
    try:
        return Path(find_in_ancestors(start, path_end))
    except FileNotFoundError:
        if warn:
            warnings.warn(f'Cannot find directory "{path_end}" in ancestors of {__file__}.')
        return None


def _check_dir_path(path, kind=None):
    if path is None:
        raise FileNotFoundError(f'No {kind} directory provided or found.\n{__doc__}')
    if not path.exists():
        raise FileNotFoundError(f'Directory "{path}" does not exist.\n{__doc__}')
    if not path.is_dir():
        raise FileNotFoundError(f'"{path}" is not a directory.\n{__doc__}')


class _Dirs:
    @cached_property
    def data(self):
        """Optional root directory that can contain "datasets", "pretrained"
        and "experiments" directories.
        """
        path = os.environ.get("VIDLU_DATA", None)
        return None if path is None else Path(path)

    @cached_property
    def datasets(self):
        """List of directories that contain datasets"""
        datasets = [self._get_path_from_env_or_find_in_data_dir("datasets"), _find('datasets'),
                    _find('datasets', start='tmp/_')]
        datasets = [x for x in datasets if x is not None]
        _check_dir_path(None if len(datasets) == 0 else datasets[0], "datasets")
        return datasets

    @cached_property
    def pretrained(self):
        """Pretrained parameters"""
        pretrained = self._get_path_from_env_or_find_in_data_dir("pretrained")
        _check_dir_path(pretrained, "pretrained")
        return pretrained

    @cached_property
    def experiments(self):
        """Cache and experimental results/states"""
        experiments = self._get_path_from_env_or_find_in_data_dir("experiments")
        _check_dir_path(experiments, "experiments")
        return experiments

    @cached_property
    def cache(self):
        """Various cache"""
        cache = self._get_path_from_env_or_find_in_data_dir("cache")
        _check_dir_path(cache, "cache")
        return cache

    @cached_property
    def saved_states(self):
        """Experimental results or incomplete training states"""
        saved_states = self.experiments / 'states'
        saved_states.mkdir(exist_ok=True)
        return saved_states

    def _get_path_from_env_or_find_in_data_dir(self, dir_name: str):
        path = self.data / dir_name if self.data is not None else (
                os.environ.get(f"VIDLU_{dir_name.upper()}", None)
                or _find(f"data/{dir_name}", warn=True))
        return None if path is None else Path(path)


_dirs = _Dirs()


def __getattr__(name):
    return getattr(_dirs, name)


def __dir__():
    return "data", "datasets", "pretrained", "experiments", "cache", "saved_states"
