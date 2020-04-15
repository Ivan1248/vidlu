import inspect
import os
from pathlib import Path
import dataclasses as dc

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
        info = _ds_to_info[name]
        subsets = info.cls.subsets
        path_args = [vup.find_in_directories(self.datasets_dirs, info.path)] if info.path else []
        if len(info.cls.subsets) == 0:
            subsets = ['all']
            load = lambda s: info.cls(*path_args, **{**info.kwargs, **kwargs})
        else:
            load = lambda s: info.cls(*path_args, s, **{**info.kwargs, **kwargs})
        splits = getattr(info.cls, 'splits', _default_splits)
        return PartedDataset({s: load(s) for s in subsets}, splits)
