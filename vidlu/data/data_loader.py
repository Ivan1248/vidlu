from functools import partialmethod
from .misc import default_collate

import torch.utils.data as tud


class DataLoader(tud.DataLoader):
    """DataLoader class with support for `Record`-typed examples."""
    __init__ = partialmethod(tud.DataLoader.__init__, collate_fn=default_collate)


class BatchTuple(tuple):
    """The type of `ZipDataLoader` outputs."""
    pass


class ZipDataLoader:
    """A data loader that produces tuples of batches from multiple data loaders"""
    __slots__ = '_data_loaders'

    def __init__(self, *data_loaders):
        self._data_loaders = data_loaders

    def __iter__(self):
        return map(BatchTuple, zip(*self._data_loaders))

    def __len__(self):
        return min(len(d) for d in self._data_loaders)
