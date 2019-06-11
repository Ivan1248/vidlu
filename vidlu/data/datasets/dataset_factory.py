import inspect
from argparse import Namespace

from . import datasets
from .datasets import *
from .. import PartedDataset


def _info(cls, path=None, default_kwargs=None):
    return Namespace(cls=cls, path=path, kwargs=default_kwargs or dict())


_ds_to_info = {k.lower(): _info(v, path=getattr(v, 'default_dir', None))
               for k, v in vars(datasets).items()
               if inspect.isclass(v) and issubclass(v, Dataset) and v is not Dataset}

_default_parts = ['all', 'trainval', 'train', 'val', 'test']
_default_splits = {
    'all': (('trainval', 'test'), 0.8),
    'trainval': (('train', 'val'), 0.8),
}


class DatasetFactory:
    def __init__(self, datasets_dir):
        self.datasets_dir = Path(datasets_dir)

    def __call__(self, name: str, **kwargs):
        name = name.lower()
        if name not in _ds_to_info:
            raise KeyError(f'No dataset has the name "{name}".')
        info = _ds_to_info[name]
        subsets = info.cls.subsets
        path_args = [self.datasets_dir / info.path] if info.path else []
        if len(info.cls.subsets) == 0:
            subsets = ['all']
            load = lambda s: info.cls(*path_args, **{**info.kwargs, **kwargs})
        else:
            load = lambda s: info.cls(*path_args, s, **{**info.kwargs, **kwargs})
        splits = getattr(info.cls, 'splits', _default_splits)
        return PartedDataset({s: load(s) for s in subsets}, splits)
