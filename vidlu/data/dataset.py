"""
Dataset with transformations that create new dataset objects.

Dataset objects should be considered immutable.
"""

import itertools
import logging
import os
import pickle
import typing as T
from argparse import Namespace
from collections import abc
import warnings
import multiprocessing
import datetime as dt
from pathlib import Path
import shutil
from enum import Enum
from functools import cached_property
import dataclasses as dc

import numpy as np
import torch
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm, trange
from typeguard import typechecked
from torchvision.datasets.utils import download_and_extract_archive

import vidlu.utils.path as vup
from vidlu.utils.misc import Stopwatch, slice_len, query_user, pickle_sizeof
from vidlu.utils.path import to_valid_path
from vidlu.utils.storage import DefaultCompressor

from .record import Record, DictRecord, LazyField


# Helpers ######################################################################

def _compress_indices(indices, max):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if max <= dtype(-1):
            return np.array(indices, dtype=dtype)
    return indices


def _subset_hash(indices):
    return hash(tuple(indices)) % 16 ** 5


# Dataset ######################################################################

class StandardDownloadableDatasetMixin:
    def download_if_necessary(self, data_dir, subdir=None):
        download_dir = data_dir / subdir if subdir is not None else data_dir
        if self.download_required(download_dir):
            if not query_user(f'{type(self).__name__} requires downloading data. Proceed?', "y",
                              timeout=10):
                raise FileNotFoundError(f"The required data does not exist in {data_dir}.")
            try:
                self.download(data_dir)
            except Exception as e:
                if download_dir.exists():
                    shutil.rmtree(download_dir)
                raise e

    def download_required(self, check_dir):
        return not check_dir.exists()

    def download(self, data_dir, remove_archive=True):
        if "url" not in self.info:
            raise TypeError(f"Dataset downloading not available for {type(self).__name__}.")
        url = self.info["url"]
        filename = Path(url).name
        download_and_extract_archive(url, data_dir.parent, filename=filename,
                                     md5=self.info.get("md5", None),
                                     remove_finished=remove_archive)
        # (data_dir.parent / filename).rename(data_dir)


class SeqChange(Enum):
    Order = "order"
    Removal = "removal"
    Repeat = "repeat"
    Addition = "addition"


class DataChange:
    SeqChange = SeqChange


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
    data_change: T.Union[bool, T.Sequence[T.Union[str, SeqChange]]] = None
    info_change: T.Union[bool, T.Sequence[str]] = None

    def __repr__(self):
        return self.name


def make_change_info(dataset, name, data_change=None, info_change=None):
    if data_change is None:
        data_change = dataset.data is None or dataset.get_example != Dataset.get_example
    if info_change is None:
        info_change = dataset.info != getattr(dataset.data, "info", dataset.info)
    return ChangeInfo(name, data_change, info_change)


class Dataset(abc.Sequence, StandardDownloadableDatasetMixin):
    """An abstract class representing a Dataset.

    All subclasses should override ``__len__``, that provides the size of the
    dataset, and ``__getitem__`` for supporting integer indexing with indexes
    from {0 .. len(self)-1}.
    """

    def __init__(self, *, name: str = None, subset: str = None, data=None, info: T.Mapping = None,
                 data_change=None, info_change=None):
        self.name = name or getattr(data, 'name', type(self).__name__)
        if subset is not None:
            self.name += f'-{subset}'
            self.subset = subset
        self.info = DictRecord(info or getattr(self, 'info', None) or getattr(data, 'info', dict()))
        self.data = data
        self.change_info = make_change_info(self, name=self.name, data_change=data_change,
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

    def __getitem__(self, idx, field=None):
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
            ds = SubrangeDataset(self, idx)
        elif isinstance(idx, (list, np.ndarray)):
            ds = SubDataset(self, idx)
        else:
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}.")
            d = self.get_example(idx)
            return d if filter_fields is None else filter_fields(d)
        if filter_fields is not None:
            func_name = f"[{field}]"
            ds = ds.map(filter_fields, func_name=func_name)
        return ds

    def __len__(self):  # This can be overridden
        return len(self.data)

    def __repr__(self):
        return f'Dataset(identifier="{self.identifier}", info={self.info})'

    def __add__(self, other):
        return self.join(other)

    def get_example(self, idx):  # This can be overridden
        return self.data[idx]

    def example_size(self, sample_count):
        return pickle_sizeof([r for r in self.permute()[:sample_count]]) // sample_count

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

    def info_cache_hdd(self, name_to_func, directory, recompute=False, **kwargs):
        """Computes, adds, and caches dataset.info attributes. .

        It can be useful to automatically
        cache preprocessed data without modifying the original dataset and make
        data loading faster.

        Args:
            name_to_func: A mapping from names to functions computing attributes
                to be stored in dataset.info, e.g.
                `setattr(dataset.info, name, func())` for a `{name: func}` mapping.
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
        return SubDataset(self, indices, subset=f'filter({func_name})', **kwargs)

    def filter_split(self, predicates, *, func_names=None, **kwargs):
        """
        Splits the dataset indices into disjoint subsets matching predicates.
        """
        indiceses = self._multi_matching_indices(predicates)
        func_names = func_names or [f'{_subset_hash(indices):x}' for indices in indiceses]
        if isinstance(func_names, str):
            func_names = [f"{func_names}_{i}" for i in range(len(predicates) + 1)]
        return [
            SubDataset(self, indices, subset=f'filter({func_name})', **kwargs)
            for indices, func_name in zip(indiceses, func_names)]

    def map(self, func, *, func_name=None, unpack=False, **kwargs):
        """Creates a dataset with elements transformed with `func`."""
        return MapDataset(self, func, func_name=func_name, unpack=unpack, **kwargs)

    def map_unpack(self, func, *, func_name=None, **kwargs):
        """Creates a dataset with elements transformed with `func`.

        Elements are unpacked into function arguments using "*".
        """
        return MapDataset(self, func, func_name=func_name, unpack=True, **kwargs)

    def map_fields(self, field_to_func, *, func_name=None, **kwargs):
        """Creates a dataset with each element transformed with its function.

        ds.map_fields(dict(x1=func1, ..., xn=funcn)) does the same as
        ds.map(lambda r: Record(x1=func1(r.x_1), ..., xn=funcn(r.xn), x(n+1)=identity, ...))

        It is useful when using multiprocessing, which uses Pickle, and Pickle
        doesn't support pickling of lambdas.
        """
        return self.map(FieldsMap(field_to_func), func_name=func_name, **kwargs)

    def enumerate(self):
        return EnumerateDataset(self)

    def permute(self, seed=53, **kwargs):
        """Creates a permutation of the dataset."""
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        return SubDataset(self, indices, subset=F"permute({seed})",
                          data_change=[SeqChange.Order], **kwargs)

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
        clear_hdd_cache(ds.data)
    elif isinstance(ds.data, T.Sequence):
        for ds_ in ds.data:
            if isinstance(ds_, Dataset):
                clear_hdd_cache(ds_)


def clean_up_dataset_cache(cache_dir, max_time_since_access: dt.timedelta):
    to_delete = []
    for dir in Path(cache_dir).iterdir():
        file = next(dir.iterdir(), None)
        if file is None or (vup.time_since_access(file) > max_time_since_access):
            to_delete.append(dir)
    if len(to_delete) > 0:
        for dir in tqdm(to_delete,
                        desc=f"Cleaning up dataset cache unused for {max_time_since_access}."):
            shutil.rmtree(dir)


@typechecked
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
        return self.func(*r) if self.unpack else self.func(r)


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


class CacheDataset(Dataset):
    __slots__ = ("_cache_all", "_cached_data")

    def __init__(self, dataset, max_cache_size=np.inf, **kwargs):
        cache_size = min(len(dataset), max_cache_size)
        self._cache_all = cache_size == len(dataset)
        super().__init__(name="cache" if self._cache_all else f"(0..{cache_size - 1})",
                         data=dataset, data_change=False, **kwargs)
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
        cache_path = f"{cache_dir}/{self.data_identifier}.p"
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
        super().__init__(name="cache_hdd_ram", data=data, info=dataset.info, data_change=False,
                         **kwargs)


def objects_equal(a, b):
    """pickle.dumps does not always give the same results and it seems to be more likely to give the
    same result if elements are compared instead of whole objects at once."""
    import PIL.Image as pimg
    if type(a) is not type(b) and not (isinstance(a, pimg.Image) and isinstance(b, pimg.Image)):
        return False
    if isinstance(a, (Record, T.Mapping)):
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


class HDDCacheDataset(Dataset):
    # Caches the whole dataset on HDD
    # TODO: add version check instead of consistency check?
    __slots__ = ('cache_dir', 'separate_fields', 'keys')

    def __init__(self, dataset, cache_dir, separate_fields=True, consistency_check_sample_count=1,
                 compressor_f=DefaultCompressor, **kwargs):
        if len(dataset) == 0:
            warnings.warn(f"The dataset {dataset} is empty.")
            return

        super().__init__(name='cache_hdd' + ('_s' if separate_fields else ''), data=dataset,
                         data_change=False, **kwargs)
        self.cache_dir = to_valid_path(Path(cache_dir) / self.data_identifier)
        self.separate_fields = separate_fields
        self.compressor = compressor_f()
        if separate_fields:
            if not isinstance(dataset[0], Record):
                raise ValueError(
                    f"If `separate_fields == True`, the element type must be `Record`.")
            self.keys = list(self.data[0].keys())
        os.makedirs(self.cache_dir, exist_ok=True)

        for i in range(consistency_check_sample_count):
            ii = (i + 1) * len(dataset) // (consistency_check_sample_count + 1)
            if not objects_equal(dataset[ii], self[ii]):
                warnings.warn(f"Cache of the dataset {self.data_identifier} inconsistent." +
                              " Deleting old and creating new cache.")
                self.delete_cache(keep_dir=True)
                break

    def _get_example_cache_path(self, idx, field=None):
        path = f"{self.cache_dir}/{idx}"
        return Path(f"{path}_{field}.p" if field else path + '.p')

    def _save(self, cache_path, obj):
        with open(cache_path, 'wb') as cache_file:
            cobj = self.compressor.compress(obj)
            pickle.dump(cobj, cache_file, protocol=4)

    def _load(self, cache_path):
        with open(cache_path, 'rb') as cache_file:
            cobj = pickle.load(cache_file)
            return self.compressor.decompress(cobj)

    def _get_example_or_field(self, idx, field=None, check=False):
        sw = Stopwatch()
        times = Namespace(orig_load=None, cache_save=None, cache_load=None)

        cache_path = self._get_example_cache_path(idx, field)
        if os.path.exists(cache_path):
            try:
                with sw.reset():
                    obj = self._load(cache_path)
                times.cache_load = sw.time
                return Namespace(obj=obj, was_cached=True, times=times)
            except (PermissionError, TypeError, AttributeError, EOFError, FileNotFoundError,
                    pickle.UnpicklingError, Exception) as e:
                short_e_str = str(e)
                if len(short_e_str) > 400:
                    short_e_str = short_e_str[:200] + "\n...\n" + short_e_str[-200:]
                print(f"HDDCacheDataset cache file invalid: {short_e_str}. File: {cache_path}")
                try:
                    cache_path.unlink()
                except FileNotFoundError as e:
                    print(f"HDDCacheDataset cache deletion error: {e}. File: {cache_path}")
                    pass
                return self._get_example_or_field(idx, field, check=True)

        with sw.reset():
            obj = self.data[idx]
            if field is not None:
                obj = obj[field]
        times.orig_load = sw.time

        with sw.reset():
            self._save(cache_path, obj)
        times.cache_save = sw.time

        if check:
            with sw.reset():
                self._load(cache_path)
            times.cache_load = sw.time
        return Namespace(obj=obj, was_cached=False, times=times)

    def _get_example_with_info(self, idx):
        if self.separate_fields:  # TODO: improve for small non-lazy fields
            return Record({f"{k}_": (lambda k_: lambda: self._get_example_or_field(idx, k_))(k)
                           for k in self.keys})
        return self._get_example_or_field(idx)

    def get_example(self, idx):
        if self.separate_fields:  # TODO: improve for small non-lazy fields
            return Record({f"{k}_": (lambda k_: lambda: self._get_example_or_field(idx, k_).obj)(k)
                           for k in self.keys})
        return self._get_example_or_field(idx).obj

    def delete_cache(self, keep_dir=False):
        if keep_dir:
            for x in self.cache_dir.iterdir():
                if x.is_dir():
                    shutil.rmtree(x)
                else:
                    try:
                        x.unlink()
                    except FileNotFoundError as e:
                        print(f"HDDCacheDataset.delete_cache: file already deleted: {x}")
                        pass
        else:
            shutil.rmtree(self.cache_dir)


class CacheIfFasterDataset(Dataset):
    def __init__(self, cache_dataset_f, cache_dir):
        super().__init__(data=cache_dataset_f(), name='cache_if_faster', data_change=False)

    def get_example(self, idx):
        return self.data[idx]


# class InfoCacheDataset(Dataset):  # lazy
#     def __init__(self, dataset, name_to_func, **kwargs):
#         self.names_str = ', '.join(name_to_func.keys())
#         self.initialized = multiprocessing.Value('i', 0)  # must be before super
#         info = NameDict(dataset.info or kwargs.get('info', dict()))
#         self._info = None
#         super().__init__(name=f"info_cache({self.names_str})", data=dataset, info=info,
#                          data_change=False, info_change=list(name_to_func), **kwargs)
#         self.name_to_func = name_to_func
#         self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
#         self._logger.addHandler(logging.NullHandler())
#
#     @property
#     def info(self):
#         if not self.initialized.value:
#             self._update_info_cache()
#         return self._info
#
#     @info.setter
#     def info(self, value):
#         """This is called by the base initializer and (unnecessarily) by pickle
#         if the object is shared between processes."""
#         self._info = value
#
#     def _compute(self):
#         self._logger.info(f"{type(self).__name__}: computing/loading {self.names_str} for"
#                           + f" {self.identifier}")
#         info_cache = dict()
#         for n, f in self.name_to_func.items():
#             info_cache[n] = f(self.data)
#         return info_cache
#
#     def _update_info_cache(self):
#         with self.initialized.get_lock():
#             if not self.initialized.value:  # lazy
#                 if any(k not in self._info for k in self.name_to_func):
#                     self._info.update(self._compute())
#                 self.initialized.value = True


class InfoCacheDataset(Dataset):  # lazy
    def __init__(self, dataset, name_to_func, **kwargs):
        self.names_str = ', '.join(name_to_func.keys())
        self.initialized = {k: multiprocessing.Value('i', 0)
                            for k in name_to_func}  # must be before super
        info = DictRecord(dataset.info or kwargs.get('info', dict()),
                          **{k: LazyField(f) for k, f in name_to_func.items()})  # lazyness support

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


class HDDInfoCacheDataset(InfoCacheDataset):  # TODO
    def __init__(self, dataset, name_to_func, cache_dir, recompute=False, simplify_dataset=None,
                 **kwargs):
        self.cache_dir = Path(cache_dir) / "info_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        check_ds = None if simplify_dataset is None else simplify_dataset(dataset)
        name_to_func = {
            k: HDDCache(dataset, func,
                        cache_file=self.cache_dir / to_valid_path((dataset.identifier + k)),
                        recompute=recompute, check_dataset=check_ds)
            for k, func in name_to_func.items()}
        super().__init__(dataset, name_to_func, **kwargs)


class SubDataset(Dataset):
    def __init__(self, dataset, indices: T.Union[T.Sequence, T.Callable], subset=None,
                 **kwargs):
        # convert indices to smaller int type if possible
        if isinstance(indices, slice):
            self._len = len(dataset)
            start, stop, step = indices.indices(slice_len(indices, self._len))
            self._get_index = lambda i: start + step * i
            choice_name_ = f"[{start}:{stop}:{step if step != 0 else ''}]"
        else:
            self._len = len(indices)
            indices = _compress_indices(indices, len(dataset))
            self._get_index = lambda i: indices[i]
            choice_name_ = f"[indices_{_subset_hash(indices):x}]"
        super().__init__(name=subset or choice_name_, data=dataset,
                         data_change=kwargs.pop("data_change", [SeqChange.Removal]), **kwargs)

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
        super().__init__(name=f"[{start}..{stop}" + ("]" if step == 1 else f";{step}]"),
                         data=dataset, data_change=[SeqChange.REMOVAL], **kwargs)

    def get_example(self, idx):
        return self.data[self.start + self.step * idx]

    def __len__(self):
        return self._len


class RepeatDataset(Dataset):
    __slots__ = ("number_of_repeats",)

    def __init__(self, dataset, number_of_repeats, **kwargs):
        super().__init__(name=f"repeat({number_of_repeats})", data=dataset,
                         data_change=[SeqChange.Repeat], **kwargs)
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
        data_change = [SeqChange.Order, SeqChange.Removal, SeqChange.Repeat] if replace else [
            SeqChange.Order]
        super().__init__(name=f"sample{'_r' if replace else ''}({args})", data=dataset,
                         data_change=data_change, **kwargs)
        self._len = length or len(dataset)

    def get_example(self, idx):
        return self.data[self._indices(idx)]

    def __len__(self):
        return self._len
