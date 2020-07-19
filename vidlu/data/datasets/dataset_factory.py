import inspect
import os
from pathlib import Path
import dataclasses as dc
import warnings

from . import datasets
from .datasets import Dataset
from .. import PartedDataset

import vidlu.utils.path as vup


@dc.dataclass
class _DatasetInfo:
    cls: type(Dataset)  # dataset class (factory)
    path: os.PathLike = None  # path to the dataset directory (if not synthetic)
    kwargs: dict = dc.field(default_factory=dict)  # arguments to always be given to cls


_ds_to_info = {k.lower(): _DatasetInfo(v, path=getattr(v, 'default_dir', None))
               for k, v in vars(datasets).items()
               if inspect.isclass(v) and issubclass(v, Dataset) and v is not Dataset}

_default_parts = ['all', 'trainval', 'train', 'val', 'test']
_default_splits = {
    'all': (('trainval', 'test'), 0.8),
    'trainval': (('train', 'val'), 0.8),
}


class DatasetFactory:
    def __init__(self, datasets_dir_or_dirs):
        if isinstance(datasets_dir_or_dirs, os.PathLike):
            datasets_dir_or_dirs = [datasets_dir_or_dirs]
        self.datasets_dirs = list(map(Path, datasets_dir_or_dirs))

    def __call__(self, name: str, **kwargs):
        name = name.lower()
        if name not in _ds_to_info:
            raise KeyError(f'No dataset has the name "{name}".')
        ds_info = _ds_to_info[name]
        subsets = ds_info.cls.subsets
        try:
            path_args = [vup.find_in_directories(self.datasets_dirs, ds_info.path)] \
                if ds_info.path else []
        except FileNotFoundError as e:
            warnings.warn(f"{ds_info.path} directory for {ds_info.cls} not found in any of "
                          + f"{[str(p) for p in self.datasets_dirs]}")
            path_args = [self.datasets_dirs[0] / ds_info.path]
        if len(ds_info.cls.subsets) == 0:
            subsets = ['all']
            load = lambda s: ds_info.cls(*path_args, **{**ds_info.kwargs, **kwargs})
        else:
            load = lambda s: ds_info.cls(*path_args, s, **{**ds_info.kwargs, **kwargs})
        splits = getattr(ds_info.cls, 'splits', _default_splits)
        return PartedDataset({s: load(s) for s in subsets}, splits)
