import secrets
from collections.abc import Sequence
from typing import Dict

import numpy as np
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from ...utils.misc import cached_property

from .. import Record, LazySeq


class Dataset(Sequence):
    """An abstract class representing a Dataset as a "structure of arrays".

    All subclasses should override ``__len__``, that provides the size of the
    dataset, and ``__getitem__`` for supporting integer indexing with indexes
    from {0 .. len(self)-1}.
    """
    subsets = None

    @classmethod
    def subset_hash(cls, indices):
        return hash(tuple(indices)) % 16 ** 5

    def __init__(self, data: Dict[str, Sequence], info: dict):
        assert not hasattr(self, 'name') and not hasattr(self, 'info')
        info = info or dict()
        info["name"] = info.get("name", self.__class__.__name__ + secrets.token_hex(4))
        info["transforms"] = info.get("transforms", [])
        self.data, self.info = data, info

    def get_example(self, idx):  # This can be overriden
        return Record({f"{k}_": lambda: d[idx] for k, d in self.data})

    @cached_property
    def example(self):
        return self[0]

    def __getitem__(self, idx, *args):
        assert len(args) <= 1
        fields = args[0] if len(args) > 0 else self.data.keys()
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            tr = f"[{start}:{stop}" + ("]" if step == 1 else f":{step}]")
        elif isinstance(idx, (list, tuple, np.ndarray)):
            tr = f"[indices_{Dataset.subset_hash(idx)}]"
        else:
            return Record(dict([(k, self.data[idx]) if type(self.data[k]) is LazySeq else
                                (f"{k}_", lambda: self.data[idx]) for k in fields]))
        return Dataset({k: self.data[k][idx] for k in fields},
                       dict(self.info, transforms=self.info["transforms"] + [tr]))

    def __len__(self):  # This can be overridden
        if hasattr(self, 'data'):
            return len(self.data)
        elif hasattr(self, '_getitem'):
            return self._len
        else:
            raise NotImplementedError

    def __str__(self):
        return f"Dataset(name={self.name}, info={self.info})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.join(other)

    def map(self, field_name_to_func, update=False, func_name=None, new_info=None):
        fields = func(self.example).keys

        ds = MapDataset(self, func, func_name)
        if new_info is not None:
            ds.info = new_info
        return ds

    def filter(self, func, *, func_name=None, new_info=None):
        self._print("Filtering dataset...")
        indices = np.array([i for i, d in enumerate(tqdm(self)) if func(d)])
        func_name = func_name or f'{self._subset_hash(indices):x}'
        ds = SubDataset(self, indices, f'filter({func_name})')
        if new_info is not None:
            ds.info = new_info
        self._print(
            f"  {len(ds)} of {len(self)} examples left in filtered dataset.")
        return ds

    def join_elements(self, *other):
        return JoinElementsDataset([self] + other)

    def batch(self, batch_size):
        return BatchDataset(self, batch_size)

    def unbatch(self):
        assert isinstance(self, BatchDataset)
        return self.data

    def cache(self, max_cache_size=np.inf, directory=None, chunk_size=100):
        # caches the dataset in RAM (partially or completely)
        if directory is not None:
            return HDDAndRAMCacheDataset(self, directory, chunk_size)
        return CacheDataset(self, max_cache_size)

    def cache_hdd(self, directory):
        return HDDCacheDataset(self, directory)

    def permute(self, seed=53):
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        return SubDataset(self, indices, f"permute({seed})")

    def repeat(self, number_of_repeats):
        return RepeatDataset(self, number_of_repeats)

    def sample_with_replacement(self, seed=53, new_len=None):
        return BootstrapDataset(self, seed=seed, new_len=new_len)

    def split(self, ratio: float = None, position: int = None):
        assert position or ratio
        indices = np.arange(len(self))
        pos = position or round(ratio * len(self))
        dsl = SubDataset(self, indices[:pos], name_addition=f"subset(0..{pos-1})")
        dsr = SubDataset(self, indices[pos:], name_addition=f".subset({pos}..{len(self)-1})")
        return dsl, dsr

    def join(self, other, info=None):
        if type(other) is not list:
            other = [other]
        datasets = [self] + other
        info = info or {k: v for k, v in datasets[0].info.items()
                        if all(d.info.get(k, None) == v for d in datasets[1:])}
        name = f"join[" + ",".join(x.name for x in datasets) + "]"
        return Dataset(name, info, data=ConcatDataset(datasets))

    def _print(self, *args, **kwargs):
        print(*args, f"({self.name})", **kwargs)
