from functools import partialmethod, partial
import typing as T
import numbers as num

import torch.utils.data as tud
import numpy as np
from typeguard import typechecked

from .collation import default_collate


def worker_init_fn(worker_id):
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# TODO: rename to DafaultDataLoader and leave DataLoader with collate_fn=None?
class DataLoader(tud.DataLoader):
    """DataLoader class with support for `Record`-typed examples."""
    __init__ = partialmethod(tud.DataLoader.__init__, collate_fn=default_collate, drop_last=True,
                             worker_init_fn=worker_init_fn)


class SingleDataLoader(tud.DataLoader):
    def __init__(self, dataset, num_workers=0, pin_memory=False, drop_last=False, timeout=0,
                 multiprocessing_context=None, *, prefetch_factor=None, persistent_workers=False):
        if prefetch_factor is None:
            prefetch_factor = 2 if num_workers == 0 else 4
        super().__init__(dataset, batch_size=1, collate_fn=lambda x: x, num_workers=num_workers,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                         multiprocessing_context=multiprocessing_context,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    def __iter__(self):
        for [x] in super().__iter__():
            yield x


class BatchTuple(tuple):
    """The type of `ZipDataLoader` outputs."""

    def __str__(self):
        content = ", ".join(str(type(x) for x in self))
        return f"BatchTuple({content})"


def _repeat(iterable, n):
    for _ in range(n):
        yield from iterable


@typechecked
class MultiDataLoaderBase:
    """Base class for iterating over several data loaders in lock-step.

    Each iteration yields one batch from each inner loader (as a `BatchTuple`). Loaders
    shorter than the primary one are cycled, so all keep producing until the primary is
    exhausted.

    Args:
        *data_loaders: The inner data loaders to iterate jointly.
        primary_index: Selects the loader whose length is the epoch length: an index, or
            ``'shortest'``/``'longest'`` (fewest/most batches), or ``'equal'``, which
            requires equal lengths and uses the first.
    """

    def __init__(self, *data_loaders,
                 primary_index: T.Optional[
                     T.Union[num.Integral, T.Literal['longest', 'shortest', 'equal']]] = 'shortest'):
        self.data_loaders = data_loaders
        lengths = np.array([len(d) for d in self.data_loaders])
        if primary_index == 'equal':
            if np.any(lengths != lengths[0]):
                raise RuntimeError(f"Data loaders are of the same length. Lengths={lengths}.")
            primary_index = 0
        elif not isinstance(primary_index, int):
            primary_index = (np.argmin(lengths) if primary_index == 'shortest' else
                             np.argmax(lengths) if primary_index == 'longest' else
                             0)
        self.primary_index = primary_index
        self._len = lengths[primary_index]
        self._repeats = None if self._len == min(lengths) else \
            [1 if l == self._len else int(self._len / l + 1) for l in lengths]

    def __iter__(self):
        iterators = self.data_loaders if self._repeats is None else \
            [_repeat(d, r) for d, r in zip(self.data_loaders, self._repeats)]
        return map(BatchTuple, zip(*iterators))

    def __len__(self):
        return self._len


class ZipDataLoader(MultiDataLoaderBase):
    """A data loader that produces tuples of batches from multiple data loaders"""


class CombinedDataLoader(MultiDataLoaderBase):
    """A data loader that merges one batch from each inner loader into a single batch.

    Each inner loader must yield an uncollated list of examples
    (`collate_fn=list`), so that per-loader batch sizes fix a constant per-source
    count in every merged batch (e.g. `batch_size=[8, 4]` -> 12 examples, always
    8 from the first dataset and 4 from the second). `collate_fn` combines the
    concatenated examples into one batch.
    """

    def __init__(self, *data_loaders, collate_fn=default_collate, primary_index='shortest'):
        if any(dl.collate_fn is not list for dl in data_loaders):
            raise TypeError('All inner data loaders should have collate_fn=list.')
        self.collate_fn = collate_fn
        super().__init__(*data_loaders, primary_index=primary_index)

    def __iter__(self):
        for elements in super().__iter__():
            yield self.collate_fn(sum(elements, []))
