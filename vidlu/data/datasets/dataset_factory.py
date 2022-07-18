import inspect
import os
from pathlib import Path
import warnings
import itertools
from functools import partial
from argparse import Namespace

from . import datasets
from .datasets import Dataset
import vidlu.utils.path as vup


class DatasetFactory:
    def __init__(self, datasets_dir_or_dirs, datasets_modules=(datasets,)):
        if isinstance(datasets_dir_or_dirs, os.PathLike):
            datasets_dir_or_dirs = [datasets_dir_or_dirs]
        self.datasets_dirs = list(map(Path, datasets_dir_or_dirs))

        self.name_to_ds_class = {
            k: v
            for k, v in itertools.chain(*(vars(dm).items() for dm in datasets_modules))
            if inspect.isclass(v) and issubclass(v, Dataset) and v is not Dataset}
        self.ds_name_lower_to_normal = {k.lower(): k for k in self.name_to_ds_class}

    def __call__(self, name: str, *args, **kwargs):
        if name not in self.name_to_ds_class:
            if (name_fixed := self.ds_name_lower_to_normal.get(name.lower(), None)) is not None:
                raise KeyError(f'No dataset has the name "{name}". Did you mean "{name_fixed}"?')
            raise KeyError(f'No dataset has the name "{name}".'
                           + f' Available datasets: {", ".join(self.name_to_ds_class.keys())}.')
        ds_class = self.name_to_ds_class[name]

        try:
            path_args = [vup.find_in_directories(self.datasets_dirs, ds_class.default_root)] \
                if ds_class.default_root else []
        except FileNotFoundError as e:
            warnings.warn(f"{ds_class.default_root} directory for {ds_class} not found in any of "
                          + f"{[str(p) for p in self.datasets_dirs]}")
            path_args = [self.datasets_dirs[0] / ds_class.default_root]

        return ds_class(*path_args, *args, **kwargs)

    def as_namespace(self):
        return Namespace(**{name: partial(self, name) for name in self.name_to_ds_class})
