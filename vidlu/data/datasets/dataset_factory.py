import inspect
import os
from pathlib import Path
import dataclasses as dc
import warnings
import itertools
from functools import partial

from . import datasets
from .datasets import Dataset
from vidlu.data.record import Record
import vidlu.utils.path as vup


@dc.dataclass
class _DatasetInfo:
    cls: type(Dataset)  # dataset class (factory)
    path: os.PathLike = None  # path to the dataset directory (if not synthetic)
    kwargs: dict = dc.field(default_factory=dict)  # arguments to always be given to cls


class DatasetFactory:
    def __init__(self, datasets_dir_or_dirs, datasets_modules=(datasets,)):
        if isinstance(datasets_dir_or_dirs, os.PathLike):
            datasets_dir_or_dirs = [datasets_dir_or_dirs]
        self.datasets_dirs = list(map(Path, datasets_dir_or_dirs))

        self.ds_to_info = {
            k: _DatasetInfo(v, path=getattr(v, 'default_dir', None))
            for k, v in itertools.chain(*(vars(dm).items() for dm in datasets_modules))
            if inspect.isclass(v) and issubclass(v, Dataset) and v is not Dataset}
        self.ds_name_lower_to_normal = {k.lower(): k for k in self.ds_to_info}

    def __call__(self, name: str, **kwargs):
        name = name
        if name not in self.ds_to_info:
            if (name_fixed := self.ds_name_lower_to_normal.get(name.lower(), None)) is not None:
                raise KeyError(f'No dataset has the name "{name}". Did you mean "{name_fixed}"?')
            raise KeyError(f'No dataset has the name "{name}".'
                           + f' Available datasets: {", ".join(self.ds_to_info.keys())}.')
        ds_info = self.ds_to_info[name]

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
            load = lambda s: ds_info.cls(*path_args, subset=s, **{**ds_info.kwargs, **kwargs})
        return Record(**{f"{s}_": partial(load, s) for s in subsets})
