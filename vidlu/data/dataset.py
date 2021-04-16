"""
Dataset with transformations that create new dataset objects.

Dataset objects should be considered immutable.
"""

import itertools
import logging
import os
import pickle
import typing as T
from collections import abc
import warnings
import multiprocessing
import datetime as dt
from pathlib import Path
import shutil

import numpy as np
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm, trange

import vidlu.utils.path as vup
from vidlu.utils.misc import slice_len, query_user
from vidlu.utils.collections import NameDict
from vidlu.utils.path import to_valid_path

from .record import Record
from .misc import default_collate, pickle_sizeof


# Helpers ######################################################################

def _compress_indices(indices, max):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max <= dtype(-1):
            return np.array(indices, dtype=dtype)
    return indices


def _subset_hash(indices):
    return hash(tuple(indices)) % 16 ** 5


def _split_on_common_prefix(strings):
    i = -1
    for i, (c, *others) in enumerate(zip(*strings)):
        if any(d != c for d in others):
            break
    else:
        i += 1
    return [s[:i] for s in strings], [s[i:] for s in strings]


def _split_on_common_suffix(strings):
    def reverse(string):
        return ''.join(reversed(string))

    rsuffix, rprefixes = _split_on_common_prefix([reverse(s) for s in strings])
    return [reverse(s) for s in rprefixes], reverse(rsuffix)


# Dataset ######################################################################

class Dataset(abc.Sequence):
    """An abstract class representing a Dataset.

    All subclasses should override ``__len__``, that provides the size of the
    dataset, and ``__getitem__`` for supporting integer indexing with indexes
    from {0 .. len(self)-1}.
    """

    def __init__(self, *, name: str = None, subset: str = None, modifiers=None,
                 info: T.Mapping = None, data=None):
        self.name = name or getattr(data, 'name', None) or type(self).__name__
        self.subset = subset or getattr(data, 'subset', None)
        self.modifiers = list(getattr(data, 'modifiers', [])) \
                         + ([modifiers] if isinstance(modifiers, str) else (modifiers or []))
        self.info = NameDict(info or getattr(data, 'info', dict()))
        self.data = data

    def download_if_necessary(self, data_dir):
        if not self.is_available(data_dir):
            if not query_user(f'Dataset not found in {data_dir}. Would you like to download it?'):
                raise FileNotFoundError(f"The dataset doesn't exist in {data_dir}.")
            self.download(data_dir)

    def is_available(self, data_dir):
        return data_dir.exists()

    def download(self, data_dir):
        raise TypeError(f"Dataset downloading not available for {type(self).__name__}.")

    @property
    def identifier(self):
        fn = self.name
        if self.subset:
            fn += '-' + self.subset
        if len(self.modifiers) > 0:
            fn += '.' + '.'.join(self.modifiers)
        return fn

    def __getitem__(self, idx, field=None):
        def element_fancy_index(d, key):
            if isinstance(key, (list, tuple)):
                if isinstance(d, (list, tuple)):
                    return type(d)(d[a] for a in key)
                if type(d) is dict:
                    return {k: d[k] for k in key}
            return d[key]

        filter_fields = (lambda x: element_fancy_index(x, field)) if field is not None else None

        if isinstance(idx, slice):
            ds = SubrangeDataset(self, idx)
        elif isinstance(idx, (T.Sequence, np.ndarray)):
            ds = SubDataset(self, idx)
        else:
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}.")
            d = self.get_example(idx)
            return d if filter_fields is None else filter_fields(d)
        if filter_fields is not None:
            field_str = '' if field is None else f', {field}'
            ds = ds.map(filter_fields, func_name=ds.modifiers.pop()[:-1] + field_str + ']')
        return ds

    def __len__(self):  # This can be overridden
        return len(self.data)

    def __str__(self):
        return f'Dataset(identifier="{self.identifier}", info={self.info})'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.join(other)

    def get_example(self, idx):  # This can be overridden
        return self.data[idx]

    def approx_example_size(self, sample_count=4):
        return pickle_sizeof([r for r in self.permute()[:sample_count]]) // sample_count

    def sample(self, length, replace=False, seed=53, **kwargs):
        """Creates a dataset with randomly chosen elements with or without
        replacement.
        """
        return SampleDataset(self, length=length, replace=replace, seed=seed, **kwargs)

    def cache(self, max_cache_size=np.inf, directory=None, chunk_size=100, **kwargs):
        """Caches the dataset in RAM (partially or completely)."""
        if directory is not None:
            return HDDAndRAMCacheDataset(self, directory, chunk_size, **kwargs)
        return CacheDataset(self, max_cache_size, **kwargs)

    def cache_hdd(self, directory, separate_fields=True, **kwargs):
        """Caches the dataset on the hard disk.

        It can be useful to automatically
        cache preprocessed data without modifying the original dataset and make
        data loading faster.

        Args:
            directory: The directory in which cached datasets are to be stored.
            separate_fields (bool): If True, record files are saved in separate files,
                e.g. labels are stored separately from input examples.
            unused_cleanup_time (datetime.timedelta): Time since last access for cached
                dataset that needs to pass so that it is to be deleted.
            **kwargs: additional arguments for the Dataset initializer.
        """
        return HDDCacheDataset(self, directory, separate_fields, **kwargs)

    def info_cache(self, name_to_func, **kwargs):
        """Caches the dataset in RAM.

        It can be useful to automatically
        cache preprocessed data without modifying the original dataset and make
        data loading faster.

        Args:
            name_to_func (Mapping): a mapping from cache field names to
                procedures that are to be called to lazily evaluate the fields.
            **kwargs: additional arguments for the Dataset initializer.
        """
        return InfoCacheDataset(self, name_to_func, **kwargs)

    def info_cache_hdd(self, name_to_func, directory, **kwargs):
        """Computes, adds, adn caches and caches dataset.info.cache attributes. .

        It can be useful to automatically
        cache preprocessed data without modifying the original dataset and make
        data loading faster.

        Args:
            name_to_func: A mapping from names to functions computing attributes
                to be stored in dataset.info.cache, e.g.
                `dataset.info.cache.name = func()` for a `{name: func}` mapping.
            directory: The directory in which info cache is to be stored.
            **kwargs: additional arguments for the Dataset initializer.
        """
        return HDDInfoCacheDataset(self, name_to_func, directory, **kwargs)

    def matching_indices(self, predicate, progress_bar=None):
        """Returns the indices of elements matching the predicate."""
        if not callable(predicate) and isinstance(predicate, T.Sequence):
            return self._multi_matching_indices(predicate, progress_bar)
        if progress_bar:
            return (i for i, d in enumerate(progress_bar(self)) if predicate(d))
        return (i for i, d in enumerate(self) if predicate(d))

    def filter(self, predicate, *, func_name=None, progress_bar=None, **kwargs):
        """Creates a dataset containing only the elements for which `func`
        evaluates to True.
        """
        indices = np.array(list(self.matching_indices(predicate, progress_bar=progress_bar)))
        func_name = func_name or f'{_subset_hash(indices):x}'
        return SubDataset(self, indices, modifiers=f'filter({func_name})', **kwargs)

    def filter_split(self, predicates, *, func_names=None, **kwargs):
        """
        Splits the dataset indices into disjoint subsets matching predicates.
        """
        indiceses = self._multi_matching_indices(predicates)
        func_names = func_names or [f'{_subset_hash(indices):x}' for indices in indiceses]
        if isinstance(func_names, str):
            func_names = [f"{func_names}_{i}" for i in range(len(predicates) + 1)]
        return [SubDataset(self, indices, modifiers=f'filter({func_name})', **kwargs)
                for indices, func_name in zip(indiceses, func_names)]

    def map(self, func, *, func_name=None, **kwargs):
        """Creates a dataset with elements transformed with `func`."""
        return MapDataset(self, func, func_name=func_name, **kwargs)

    def map_fields(self, field_to_func, *, func_name=None, **kwargs):
        """Creates a dataset with each element transformed with its function.

        ds.map_fields(dict(x1=func1, ..., xn=funcn)) does the same as
        ds.map(lambda r: Record(x1=func1(r.x_1), ..., xn=funcn(r.xn), x(n+1)=identity, ...))

        It is useful when using multiprocessing, which uses pickling, which
        doesn't support pickling of lambdas.
        """
        return self.map(FieldsMap(field_to_func), func_name=func_name, **kwargs)

    def enumerate(self, offset=0):
        return EnumeratedDataset(self, offset=offset)

    def permute(self, seed=53, **kwargs):
        """Creates a permutation of the dataset."""
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        return SubDataset(self, indices, modifiers=F"permute({seed})", **kwargs)

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
        name = f"join(" + ",".join(x.identifier for x in datasets) + ")"
        return Dataset(name=name, info=info, data=ConcatDataset(datasets), **kwargs)
        
    def zip(self, *other, **kwargs):
        return ZipDataset([self] + list(other), **kwargs)

    @staticmethod
    def from_getitem_func(func, len, **kwargs):
        class _Data:
            def __len__(self):
                return len

            def __getitem__(self, item):
                return func(item)

        return Dataset(data=_Data(), **kwargs)

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


def clear_hdd_cache(ds):
    import inspect
    if hasattr(ds, 'cache_dir'):
        shutil.rmtree(ds.cache_dir)
        print(f"Deleted {ds.cache_dir}")
    elif isinstance(ds, MapDataset):  # lazyNormalizer
        cache_path = inspect.getclosurevars(ds.func).nonlocals['f'].__ds__.cache_path
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Deleted {cache_path}")
    if isinstance(ds.data, Dataset):
        ds.data.clear_hdd_cache()
    elif isinstance(ds.data, T.Sequence):
        for ds in ds.data:
            if isinstance(ds, Dataset):
                ds.clear_hdd_cache()


class FieldsMap:
    # TODO: use @vuf.type_checked()
    def __init__(self, field_to_func, *, mode: T.Literal['override', 'replace'] = 'override'):
        if not mode in ('override', 'replace'):
            raise ValueError(f"Invalid mode argument '{mode}' is not in {'override', 'replace'}.")
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

    def __init__(self, dataset, func=lambda x: x, func_name=None, **kwargs):
        transform = 'map' + ('_' + func_name if func_name else '')
        super().__init__(modifiers=transform, data=dataset, **kwargs)
        self.func = func

    def get_example(self, idx):
        return self.func(self.data[idx])


class EnumeratedDataset(Dataset):
    __slots__ = ("offset",)

    def __init__(self, dataset, offset=0, field_name="id", **kwargs):
        super().__init__(modifiers=f'enumerated({offset})', data=dataset, **kwargs)
        self.offset = offset
        self.field_name = field_name

    def get_example(self, idx):
        r = self.data[idx]
        return type(r)(r, **{self.field_name: idx}) if isinstance(r, T.Mapping) \
            else r + type(r)((idx,))


class ZipDataset(Dataset):
    def __init__(self, datasets, **kwargs):
        if not all(len(d) == len(datasets[0]) for d in datasets):
            raise ValueError("All datasets must have the same length.")
        super().__init__(name='zip[' + ','.join(x.identifier for x in datasets) + "]",
                         data=datasets, **kwargs)

    def get_example(self, idx):
        return tuple(d[idx] for d in self.data)

    def __len__(self):
        return len(self.data[0])


class CacheDataset(Dataset):
    __slots__ = ("_cache_all", "_cached_data")

    def __init__(self, dataset, max_cache_size=np.inf, **kwargs):
        cache_size = min(len(dataset), max_cache_size)
        self._cache_all = cache_size == len(dataset)
        modifier = "cache" if self._cache_all else f"(0..{cache_size - 1})"
        super().__init__(modifiers=modifier, data=dataset, **kwargs)
        if self._cache_all:
            self._print("Caching whole dataset...")
            self._cached_data = [x for x in tqdm(dataset)]
        else:
            self._print(
                f"Caching {cache_size}/{len(dataset)} of the dataset in RAM...")
            self._cached_data = [dataset[i] for i in trange(cache_size, desc="CacheDataset")]

    def get_example(self, idx):
        cache_hit = self._cache_all or idx < len(self._cached_data)
        return (self._cached_data if cache_hit else self.data)[idx]


class HDDAndRAMCacheDataset(Dataset):
    # Caches the whole dataset both on HDD and RAM
    __slots__ = ("cache_dir",)

    def __init__(self, dataset, cache_dir, chunk_size=100, **kwargs):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = f"{cache_dir}/{self.identifier}.p"
        data = None
        if os.path.exists(cache_path):
            try:
                self._print("Loading dataset cache from HDD...")
                with open(cache_path, 'rb') as f:
                    n = (len(dataset) - 1) // chunk_size + 1  # ceil
                    chunks = [pickle.load(f)
                              for _ in trange(n, desc="Loading dataset cache from HDD")]
                    data = list(itertools.chain(*chunks))
            except Exception as e:
                self._print(e)
                self._print("Removing invalid HDD-cached dataset...")
                os.remove(cache_path)
        if data is None:
            self._print(f"Caching whole dataset in RAM...")
            data = [x for x in tqdm(dataset, desc="Caching whole dataset in RAM")]

            self._print("Saving dataset cache to HDD...")

            def to_chunks(data):
                chunk = []
                for x in data:
                    chunk.append(x)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                yield chunk

            with open(cache_path, 'wb') as f:
                # examples pickled in chunks because of memory constraints
                for x in tqdm(to_chunks(data), desc="Saving dataset cache to HDD"):
                    pickle.dump(x, f, protocol=4)
        super().__init__(modifiers="cache_hdd_ram", info=dataset.info, data=data, **kwargs)


def clean_up_dataset_cache(cache_dir, max_time_since_access: dt.timedelta):
    to_delete = []
    for dir in Path(cache_dir).iterdir():
        file = next(dir.iterdir(), None)
        if file is None or (vup.time_since_access(file) > max_time_since_access):
            to_delete.append(dir)
    for dir in tqdm(to_delete,
                    desc=f"Cleaning up dataset cache unused for {max_time_since_access}."):
        shutil.rmtree(dir)


class HDDCacheDataset(Dataset):
    # Caches the whole dataset on HDD
    __slots__ = ('cache_dir', 'separate_fields', 'keys')

    def __init__(self, dataset, cache_dir, separate_fields=True, consistency_check_sample_count=4,
                 **kwargs):
        super().__init__(modifiers='cache_hdd' + ('_s' if separate_fields else ''),
                         data=dataset, **kwargs)
        self.cache_dir = to_valid_path(Path(cache_dir) / self.identifier)
        self.separate_fields = separate_fields
        if len(dataset) == 0:
            warnings.warn(f"The dataset {dataset} is empty.")
            return
        if separate_fields:
            if not isinstance(dataset[0], Record):
                raise ValueError(
                    f"If `separate_fields == True`, the element type must be `Record`.")
            self.keys = list(self.data[0].keys())
        os.makedirs(self.cache_dir, exist_ok=True)
        for i in range(consistency_check_sample_count):
            ii = i * len(dataset) // consistency_check_sample_count
            if pickle.dumps(dataset[ii]) != pickle.dumps(self[ii]):
                warnings.warn(f"Cache of the dataset {self.identifier} inconsistent." +
                              " Deleting old and creating new cache.")
                self.delete_cache()
                os.makedirs(self.cache_dir, exist_ok=False)
                break

    def _get_example_cache_path(self, idx, field=None):
        path = f"{self.cache_dir}/{idx}"
        return f"{path}_{field}.p" if field else path + '.p'

    def _get_example_or_field(self, idx, field=None):
        cache_path = self._get_example_cache_path(idx, field)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as cache_file:
                    return pickle.load(cache_file)
            except (PermissionError, TypeError, EOFError):
                os.remove(cache_path)
        example = self.data[idx]
        if field is not None:
            example = example[field]
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(example, cache_file, protocol=4)
        return example

    def get_example(self, idx):
        if self.separate_fields:  # TODO: improve for small non-lazy fields
            return Record({f"{k}_": (lambda k_: lambda: self._get_example_or_field(idx, k_))(k)
                           for k in self.keys})
        return self._get_example_or_field(idx)

    def delete_cache(self):
        shutil.rmtree(self.cache_dir)


class InfoCacheDataset(Dataset):  # TODO
    # TODO: rename with "lazy"
    def __init__(self, dataset, name_to_func, verbose=True, **kwargs):
        self.names_str = ', '.join(name_to_func.keys())
        modifier = f"info_cache({self.names_str})"
        self.initialized = multiprocessing.Value('i', 0)  # must be before super
        info = NameDict(dataset.info or kwargs.get('info', dict()))
        info.cache = NameDict(info.get('cache', NameDict()))
        self._info = None
        super().__init__(modifiers=modifier, data=dataset, info=info, **kwargs)
        self.name_to_func = name_to_func
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._logger.addHandler(logging.NullHandler())

    @property
    def info(self):
        if not self.initialized.value:
            self._update_info_cache()
        return self._info

    @info.setter
    def info(self, value):
        """This is called by the base initializer and (unnecessarily) by pickle
        if the object is shared between processes."""
        self._info = value

    def _get_info_cache(self):
        self._logger.info(f"{type(self).__name__}: computing/loading {self.names_str} for"
                          + f" {self.data.identifier}")
        info_cache = dict()
        for n, f in self.name_to_func.items():
            info_cache[n] = f(self.data)
        return info_cache

    def _update_info_cache(self):
        with self.initialized.get_lock():
            if not self.initialized.value:  # lazy
                if any(k not in self._info.cache for k in self.name_to_func):
                    self._info.cache.update(self._get_info_cache())
                self.initialized.value = True


class HDDInfoCacheDataset(InfoCacheDataset):  # TODO
    def __init__(self, dataset, name_to_func, cache_dir, **kwargs):
        super().__init__(dataset, name_to_func, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_file = to_valid_path(self.cache_dir / self.identifier / 'info_cache.txt')
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_info_cache(self):
        if self.cache_file.exists():
            try:  # load
                with self.cache_file.open('rb') as file:
                    return pickle.load(file)
            except (PermissionError, TypeError):
                self.cache_file.unlink()
                raise
        else:
            info_cache = super()._get_info_cache()
            try:  # store
                with self.cache_file.open('wb') as file:
                    pickle.dump(info_cache, file)
            except (PermissionError, TypeError):
                self.cache_file.unlink()
                raise
            return info_cache


class SubDataset(Dataset):
    __slots__ = ("_len", "_get_index")

    def __init__(self, dataset, indices: T.Union[T.Sequence, T.Callable], **kwargs):
        # convert indices to smaller int type if possible
        if isinstance(indices, slice):
            self._len = len(dataset)
            start, stop, step = indices.indices(slice_len(indices, self._len))
            self._get_index = lambda i: start + step * i
            modifier = f"[{start}:{stop}:{step if step != 0 else ''}]"
        else:
            self._len = len(indices)
            indices = _compress_indices(indices, len(dataset))
            self._get_index = lambda i: indices[i]
            modifier = f"[indices_{_subset_hash(indices):x}]"
        kwargs['modifiers'] = kwargs.pop('modifiers', modifier)
        super().__init__(data=dataset, **kwargs)

    def get_example(self, idx):
        return self.data[self._get_index(idx)]

    def __len__(self):
        return self._len


class SubrangeDataset(Dataset):
    __slots__ = ("start", "stop", "step", "_len")

    def __init__(self, dataset, slice_, **kwargs):
        start, stop, step = slice_.indices(len(dataset))
        self.start, self.stop, self.step = start, stop, step
        self._len = slice_len(slice_, len(dataset))
        modifier = f"[{start}..{stop}" + ("]" if step == 1 else f";{step}]")
        super().__init__(modifiers=[modifier], data=dataset, **kwargs)

    def get_example(self, idx):
        return self.data[self.start + self.step * idx]

    def __len__(self):
        return self._len


class RepeatDataset(Dataset):
    __slots__ = ("number_of_repeats",)

    def __init__(self, dataset, number_of_repeats, **kwargs):
        super().__init__(modifiers=[f"repeat({number_of_repeats})"], data=dataset, **kwargs)
        self.number_of_repeats = number_of_repeats

    def get_example(self, idx):
        return self.data[idx % len(self.data)]

    def __len__(self):
        return len(self.data) * self.number_of_repeats


class SampleDataset(Dataset):
    __slots__ = ("_indices", "_len")

    def __init__(self, dataset, length=None, replace=False, seed=53, **kwargs):
        length = length or len(dataset)
        if length > len(dataset) and not replace:
            raise ValueError("Cannot sample without replacement if `length` is larger than the "
                             + " length of the original dataset.")
        rand = np.random.RandomState(seed=seed)
        if replace:
            indices = [rand.randint(0, len(dataset)) for _ in range(len(dataset))]
        else:
            indices = rand.permutation(len(dataset))[:length]
            if length > len(dataset):
                raise ValueError("A sample without replacement cannot be larger"
                                 + " than the original dataset.")
        self._indices = _compress_indices(indices, len(dataset))
        args = f"{seed}"
        if length is not None:
            args += f",{length}"
        modifier = f"sample{'_r' if replace else ''}({args})"
        super().__init__(modifiers=modifier, data=dataset, **kwargs)
        self._len = length or len(dataset)

    def get_example(self, idx):
        return self.data[self._indices(idx)]

    def __len__(self):
        return self._len
