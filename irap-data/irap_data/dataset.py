"""
Dataset with transformations that create new dataset objects.

Dataset objects should be considered immutable.
"""

import dataclasses as dc
import logging
import multiprocessing
import pickle
import re
import typing as T
import warnings
from collections import abc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from .lazy_dict import LazyDict, Lazy


# Helpers ######################################################################

def _compress_indices(indices, max):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max <= np.array(-1).astype(dtype):
            return np.array(indices, dtype=dtype)
    return indices


def _subset_hash(indices):
    return hash(tuple(indices)) % 16 ** 5


def _to_valid_path(s):
    return Path(re.sub(r"(?u)[^\-\w.,\'\\/!#$%^&()_+=@{}\[\]]", "+", str(s).strip()))


# Dataset ######################################################################

@dc.dataclass
class ChangeInfo:
    """An object describing the kind of change between the original and the
    modified dataset.

    This is used for deciding whether a change affects data cache and info
    cache. E.g. a data cache identifier should not change if the info is
    modified..

    Args:
        name (str): The name of the change.
        data_change (bool|Sequence[str, SequenceChange]): False indicattes no
            data change. True indicates that if one does not want to express the
            exact kinds of changes (the safest option to avoid invalid data
            cache at a cost of more cache). A sequence can contain names of
            changed fields and SequenceChange values. SequenceChange values
            indicate changes of order, or removal, repeats or additions of
            examples.
        info_change (bool|Sequence[str]): Indicates info (field) changes in the
            same way that `data_change` indicates example (field) changes,
            except SequenceChange values are not supported since there is always
            a single info instance per dataset.
    """
    name: str
    data_change: bool | T.Sequence[str] = None
    info_change: bool | T.Sequence[str] = None

    def __repr__(self):
        return self.name


def make_change_info(dataset, name, data_change=None, info_change=None):
    if data_change is None:
        data_change = dataset.data is None or dataset.get_example != Dataset.get_example
    if info_change is None:
        info_change = dataset.info != getattr(dataset.data, "info", dataset.info)
    return ChangeInfo(name, data_change, info_change)


class Dataset(abc.Sequence):
    """An abstract class representing a Dataset.

    All subclasses should override ``__len__``, that provides the size of the
    dataset, and ``__getitem__`` for supporting integer indexing with indexes
    from {0 .. len(self)-1}.
    """

    def __init__(self, *, name: str = None, subset: str = None, data: T.Optional[T.Sequence] = None,
                 info: T.Mapping = None, data_change=None, info_change=None):
        self.name = name or getattr(data, 'name', type(self).__name__)
        if subset is not None:
            self.name += f'-{subset}'
            self.subset = subset
        self.info = LazyDict(info or getattr(self, 'info', None) or getattr(data, 'info', dict()))
        self.data = data
        self.change_info = make_change_info(self, name=self.name,
                                            data_change=subset if data_change is None else data_change,
                                            info_change=info_change)

    @property
    def changes(self):
        if hasattr(self.data, "changes"):
            return [*self.data.changes, self.change_info]
        return [self.change_info]

    @property
    def identifier(self):
        return ".".join(c.name for c in self.changes)

    @property
    def data_identifier(self):
        return ".".join(c.name for c in self.changes if c.data_change is not False)

    def _getitem(self, idx, field=None, **kwargs):
        if isinstance(idx, tuple):
            idx, [field] = idx[0], idx[1:]

        def element_fancy_index(r, key):
            if isinstance(r, (dict, list, tuple)) and isinstance(key, list):
                if isinstance(r, (list, tuple)):
                    return type(r)(r[a] for a in key)
                if type(r) is dict:
                    return {k: r[k] for k in key}
            return r[key]

        filter_fields = (lambda x: element_fancy_index(x, field)) if field is not None else None

        if isinstance(idx, slice):
            ds = SubrangeDataset(self, idx, **kwargs)
        elif isinstance(idx, (list, np.ndarray)):
            ds = SubDataset(self, idx, **kwargs)
        else:
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}.")
            d = self.get_example(idx)
            return d if filter_fields is None else filter_fields(d)
        if filter_fields is not None:
            ds = ds.map(filter_fields, **{'func_name': f"[{field}]", **kwargs})
        return ds

    def __getitem__(self, idx, field=None):
        return self._getitem(idx, field=field)

    def __len__(self):  # This can be overridden
        return len(self.data)

    def __repr__(self):
        return f'Dataset(identifier="{self.identifier}", info={self.info})'

    def __add__(self, other):
        return self.join(other)

    def get_example(self, idx):  # This can be overridden
        return self.data[idx]

    def info_cache_hdd(self, name_to_func, directory, recompute=False, **kwargs):
        """Computes, adds, and caches dataset.info attributes on HDD.

        Args:
            name_to_func: A mapping from names to functions computing attributes
                to be stored in dataset.info.
            directory: The directory in which info cache is to be stored.
            **kwargs: additional arguments for the Dataset initializer.
        """
        return HDDInfoCacheDataset(self, name_to_func, directory, recompute=recompute, **kwargs)

    def find(self, predicate, progress_bar=None):
        """Returns the indices of elements matching the predicate."""
        if progress_bar:
            self = (tqdm if progress_bar is True else progress_bar)(self)
        return ((i, r) for i, r in enumerate(self) if predicate(r))

    def find_indices(self, predicate, progress_bar=None):
        """Returns the indices of elements matching the predicate."""
        if not callable(predicate) and isinstance(predicate, T.Sequence):
            return self._multi_matching_indices(predicate, progress_bar=progress_bar)
        return (i for i, r in self.find(predicate, progress_bar=progress_bar))

    def filter(self, predicate, *, func_name=None, progress_bar=None, **kwargs):
        """Creates a dataset containing only the elements for which `func`
        evaluates to True.
        """
        indices = np.array(list(self.find_indices(predicate, progress_bar=progress_bar)))
        func_name = func_name or f'{_subset_hash(indices):x}'
        return self._getitem(indices, subset=f'filter({func_name})', **kwargs)

    def filter_split(self, predicates, *, func_names=None, **kwargs):
        """
        Splits the dataset indices into disjoint subsets matching predicates.
        """
        indiceses = self._multi_matching_indices(predicates)
        func_names = func_names or [f'{_subset_hash(indices):x}' for indices in indiceses]
        if isinstance(func_names, str):
            func_names = [f"{func_names}_{i}" for i in range(len(predicates) + 1)]
        return [
            self._getitem(indices, subset=f'filter({func_name})', **kwargs)
            for indices, func_name in zip(indiceses, func_names)]

    def filter_fields(self, fields, **kwargs):
        """Creates a dataset with fields filtered to the given list.
        """
        return self.map(lambda r: r[fields], func_name=f'filter_fields' + ','.join(fields),
                        **kwargs)

    def map(self, func, *, func_name=None, unpack=False, **kwargs):
        """Creates a dataset with elements transformed with `func`."""
        return MapDataset(self, func, func_name=func_name, unpack=unpack, **kwargs)

    def map_unpack(self, func, *, func_name=None, **kwargs):
        """Creates a dataset with elements transformed with `func`.

        Elements are unpacked into function arguments using "*".
        """
        return MapDataset(self, func, func_name=func_name, unpack=True, **kwargs)

    def map_fields(self, field_to_func, *, func_name=None, **kwargs):
        """Creates a dataset with each element transformed with its function."""
        return self.map(FieldsMap(field_to_func), func_name=func_name, **kwargs)

    def enumerate(self):
        return EnumerateDataset(self)

    def permute(self, seed=53, **kwargs):
        """Creates a permutation of the dataset."""
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        return self._getitem(indices, subset=F"permute({seed})", **kwargs)

    def repeat(self, number_of_repeats, **kwargs):
        """Creates a dataset with `number_of_repeats` times the length of the
        original dataset so that every `number_of_repeats` an element is
        repeated.
        """
        return RepeatDataset(self, number_of_repeats, **kwargs)

    def split(self, ratio: float = None, index: int = None):
        if (ratio is None) == (index is None):
            raise ValueError("Either ratio or position needs to be specified.")
        if isinstance(ratio, int):
            raise ValueError("ratio should be a float. Did you intend `index={ratio}`?")
        index = index or round(ratio * len(self))
        return self[:index], self[index:]

    def join(self, *other, **kwargs):
        datasets = [self] + list(other)
        info = kwargs.pop('info', datasets[0].info)
        return Dataset(name=f"join(" + ",".join(x.identifier for x in datasets) + ")", info=info,
                       data=ConcatDataset(datasets), **kwargs)

    def zip(self, *other, **kwargs):
        return ZipDataset([self] + list(other), **kwargs)

    def sample(self, length, replace=False, seed=53, **kwargs):
        """Creates a dataset with randomly chosen elements with or without
        replacement.
        """
        return SampleDataset(self, length=length, replace=replace, seed=seed, **kwargs)

    def _multi_matching_indices(self, predicates, progress_bar=None):
        """Splits the dataset indices into disjoint subsets matching predicates.
        """
        progress_bar = progress_bar or (lambda x: x)
        indiceses = [[] for _ in range(len(predicates) + 1)]
        for i, d in enumerate(progress_bar(self)):
            for j, p in enumerate(predicates):
                if p(d):
                    indiceses[j].append(i)
                    break
                indiceses[-1].append(i)
        return indiceses

    def _print(self, *args, **kwargs):
        print(*args, f"({self.identifier})", **kwargs)


class FieldsMap:
    def __init__(self, field_to_func, *, mode: T.Literal['override', 'replace'] = 'override'):
        self.field_to_func = field_to_func
        self.mode = mode

    def __call__(self, r):
        if self.mode == 'override':
            return type(r)(r, **{k: f(r[k]) for k, f in self.field_to_func.items()})
        else:
            return type(r)(**{k: f(r[k]) for k, f in self.field_to_func.items()})


# Dataset wrappers and proxies


class MapDataset(Dataset):
    __slots__ = ("func",)

    def __init__(self, dataset, func=lambda x: x, func_name=None, unpack=False, **kwargs):
        super().__init__(
            name=f"map{'_' + func_name if func_name else ''}{'_unpack' if unpack else ''}",
            data=dataset, **kwargs)
        self.func = func
        self.unpack = unpack

    def get_example(self, idx):
        r = self.data[idx]
        if self.unpack:
            return self.func(*r) if isinstance(r, T.Sequence) else self.func(**r)
        return self.func(r)


class EnumerateDataset(Dataset):
    __slots__ = ("offset",)

    def __init__(self, dataset, **kwargs):
        super().__init__(name=f'enumerate()', data=dataset, **kwargs)

    def get_example(self, idx):
        return idx, self.data[idx]


class ZipDataset(Dataset):
    def __init__(self, datasets, strict=False, **kwargs):
        if not all(len(d) == len(datasets[0]) for d in datasets):
            raise ValueError("All datasets must have the same length.")
        self.strict = strict
        name = f"zip{'strict' if strict else ''}({','.join(x.identifier for x in datasets)})"
        super().__init__(data=datasets, name=name, **kwargs)

    def get_example(self, idx):
        return tuple(d[idx] for d in self.data)

    def __len__(self):
        return len(self.data[0]) if self.strict else min(len(d) for d in self.data)


def objects_equal(a, b):
    """pickle.dumps does not always give the same results and it seems to be more likely to give the
    same result if elements are compared instead of whole objects at once."""
    import PIL.Image as pimg
    if type(a) is not type(b) and not (isinstance(a, pimg.Image) and isinstance(b, pimg.Image)):
        return False
    if hasattr(type(a), 'keys'):
        if a.keys() != b.keys():
            return False
        return all(objects_equal(a[k], b[k]) for k in a.keys())
    elif isinstance(a, (list, tuple)):
        return all(objects_equal(ai, bi) for ai, bi in zip(a, b))
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.all(a == b)
    elif isinstance(a, pimg.Image) and isinstance(b, pimg.Image):
        return a.mode == b.mode and objects_equal(np.array(a), np.array(b))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return objects_equal(a.numpy(), b.numpy())
    elif isinstance(a, (int, float, str)):
        return a == b
    return pickle.dumps(a) == pickle.dumps(b)


class InfoCacheDataset(Dataset):  # lazy
    def __init__(self, dataset, name_to_func, **kwargs):
        self.names_str = ', '.join(name_to_func.keys())
        self.initialized = {k: multiprocessing.Value('i', 0)
                            for k in name_to_func}  # must be before super
        info = LazyDict(dataset.info or kwargs.get('info', dict()),
                        **{k: Lazy(f) for k, f in name_to_func.items()})  # laziness support

        super().__init__(name=f"info_cache({self.names_str})", data=dataset, info=info,
                         data_change=False, info_change=list(name_to_func), **kwargs)
        self.name_to_func = name_to_func
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._logger.addHandler(logging.NullHandler())


class HDDCache:
    def __init__(self, dataset, compute, cache_file, recompute=False, check_dataset=None):
        self.dataset = dataset
        self.check_dataset = check_dataset
        self.compute = compute
        self.cache_file = Path(cache_file)
        self.recompute = recompute

    def __call__(self):
        ds = self.dataset
        check = None if self.check_dataset is None else self.compute(self.check_dataset)
        if self.cache_file.exists():
            if self.recompute:
                self.cache_file.unlink()
            else:
                try:  # load
                    with self.cache_file.open('rb') as file:
                        info_cache, check_cache = pickle.load(file)
                except (PermissionError, TypeError, EOFError, AttributeError,
                        pickle.UnpicklingError, ValueError):
                    self.cache_file.unlink()
                    warnings.warn("Error loading cache. The cache file will have to be recreated.")
                else:
                    if objects_equal(check_cache, check):
                        return info_cache
                    else:
                        self.cache_file.unlink()
        info_cache = self.compute(ds)
        try:  # store
            self.cache_file.parent.mkdir(exist_ok=True)
            with self.cache_file.open('wb') as file:
                pickle.dump((info_cache, check), file)
        except (PermissionError, TypeError):
            self.cache_file.unlink()
            raise
        return info_cache


class HDDInfoCacheDataset(InfoCacheDataset):
    def __init__(self, dataset, name_to_func, cache_dir, recompute=False, simplify_dataset=None,
                 **kwargs):
        self.cache_dir = Path(cache_dir) / "info_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        check_ds = None if simplify_dataset is None else simplify_dataset(dataset)
        name_to_func = {
            k: HDDCache(dataset, func,
                        cache_file=self.cache_dir / _to_valid_path(dataset.identifier + k),
                        recompute=recompute, check_dataset=check_ds)
            for k, func in name_to_func.items()}
        super().__init__(dataset, name_to_func, **kwargs)


class SubDataset(Dataset):
    __slots__ = ('indices',)

    def __init__(self, dataset, indices: T.Union[T.Sequence, T.Callable], subset=None,
                 **kwargs):
        # convert indices to smaller int type if possible
        self.indices = _compress_indices(indices, len(dataset))
        choice_name_ = f"[indices_{_subset_hash(indices):x}]"
        super().__init__(name=subset or choice_name_, data=dataset,
                         data_change=kwargs.pop("data_change", [choice_name_]), **kwargs)

    def get_example(self, idx):
        return self.data[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class SubrangeDataset(Dataset):
    __slots__ = ("start", "stop", "step", "_len")

    def __init__(self, dataset, slice_, **kwargs):
        start, stop, step = slice_.indices(len(dataset))
        self.start, self.stop, self.step = start, stop, step
        self._len = len(range(start, stop, step))
        choice_name_ = f"[{start}:{stop}:{step if step != 0 else ''}]"
        super().__init__(name=f"[{start}..{stop}" + ("]" if step == 1 else f";{step}]"),
                         data=dataset, data_change=kwargs.pop("data_change", [choice_name_]),
                         **kwargs)

    def get_example(self, idx):
        return self.data[self.start + self.step * idx]

    def __len__(self):
        return self._len


class RepeatDataset(Dataset):
    __slots__ = ("number_of_repeats",)

    def __init__(self, dataset, number_of_repeats, **kwargs):
        name = f"repeat({number_of_repeats})"
        super().__init__(name=name, data=dataset, data_change=[name], **kwargs)
        self.number_of_repeats = number_of_repeats

    def get_example(self, idx):
        return self.data[idx % len(self.data)]

    def __len__(self):
        return len(self.data) * self.number_of_repeats


class SampleDataset(Dataset):
    __slots__ = ("_indices", "_len")

    def __init__(self, dataset, length=None, replace=False, seed=53, **kwargs):
        length = length or len(dataset)
        if length != len(dataset) and not replace:
            raise ValueError("Cannot sample without replacement if `length` is different from the"
                             + " original length.")
        rand = np.random.RandomState(seed=seed)
        if replace:
            indices = [rand.randint(0, len(dataset)) for _ in range(len(dataset))]
        else:
            indices = rand.permutation(len(dataset))[:length]
        self._indices = _compress_indices(indices, len(dataset))
        args = f"{seed}"
        if length is not None:
            args += f",{length}"
        name = f"sample{'_r' if replace else ''}({args})"
        super().__init__(name=name, data=dataset, data_change=name, **kwargs)
        self._len = length or len(dataset)

    def get_example(self, idx):
        return self.data[self._indices[idx]]

    def __len__(self):
        return self._len
