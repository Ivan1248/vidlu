from functools import partialmethod
from .misc import default_collate
from warnings import warn
import typing as T

import torch.utils.data as tud


class DataLoader(tud.DataLoader):
    """DataLoader class with support for `Record`-typed examples."""
    __init__ = partialmethod(tud.DataLoader.__init__, collate_fn=default_collate)


class BatchTuple(tuple):
    """The type of `ZipDataLoader` outputs."""
    pass


def _check_zip_data_loader_sizes(data_loaders, primary_index):
    N = len(data_loaders[primary_index if primary_index != 'equal' else 0])
    if primary_index != 'equal':
        if any(len(dl) < N for dl in data_loaders):
            warn(f"The primary dataset is smaller than a/the secondary.")
    else:
        if any(len(dl) != N for dl in data_loaders):
            warn(f"The argument {primary_index=} does not allow different numbers of batches"
                 + f" per dataset, {[len(d) for d in data_loaders]=}")


class ZipDataLoader:
    """A data loader that produces tuples of batches from multiple data loaders"""
    __slots__ = '_data_loaders', '_len'

    def __init__(self, *data_loaders,
                 primary_index: T.Optional[T.Union[int, T.Literal['equal']]] = None):
        self._data_loaders = data_loaders
        self._len = min(len(d) for d in self._data_loaders)
        if primary_index is not None:
            _check_zip_data_loader_sizes(data_loaders, primary_index)

    def __iter__(self):
        return map(BatchTuple, zip(*self._data_loaders))

    def __len__(self):
        return self._len
