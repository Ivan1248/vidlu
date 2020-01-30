"""
Dataset with transformations that create new dataset objects.

Dataset objects should be considered immutable.
"""

import itertools
import os
import pickle
from collections.abc import Sequence
from typing import Union, Callable, Mapping  #, Literal
from pathlib import Path
import shutil
import warnings
import multiprocessing

import numpy as np
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm, trange

from vidlu.utils.misc import slice_len, query_yes_no
from vidlu.utils.collections import NameDict
from vidlu.utils.path import to_valid_path
import vidlu.utils.func as vuf

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

class Dataset(Sequence):
    """An abstract class representing a Dataset.

    All subclasses should override ``__len__``, that provides the size of the
    dataset, and ``__getitem__`` for supporting integer indexing with indexes
    from {0 .. len(self)-1}.
    """
    __slots__ = ("name", "subset", "modifiers", "info", "data")
    _instance_count = 0

    def __init__(self, *, name: str = None, subset: str = None, modifiers=None,
                 info: Mapping = None, data=None):
        self.name = name or getattr(data, 'name', None)
        if self.name is None:
            if type(self) is Dataset:
                self.name = 'Dataset' + str(Dataset._instance_count)
                Dataset._instance_count += 1
            cname = type(self).__name__
            self.name = cname[:-len('Dataset')] if cname.endswith('Dataset') else cname
        self.subset = subset or getattr(data, 'subset', None)
        self.modifiers = list(getattr(data, 'modifiers', []))
        if modifiers:
            self.modifiers += [modifiers] if isinstance(modifiers, str) else modifiers
        self.info = NameDict(info or getattr(data, 'info', dict()))
        self.data = data

    def download_if_necessary(self, data_dir):
        if not data_dir.exists():
            if not query_yes_no(f'Dataset not found in {data_dir}. Would you like to download it?'):
                raise FileNotFoundError(f"The dataset doesn't exist in {data_dir}.")
            self.download(data_dir)

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
        elif isinstance(idx, (Sequence, np.ndarray)):
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

    def batch(self, batch_size, **kwargs):  # Element type is tuple
        """
        Creates a dataset where elements are grouped in `batch_size`-tuples. The
        last tuple may be smaller.
        """
        return BatchDataset(self, batch_size, **kwargs)

    def sample(self, length, replace=False, seed=53, **kwargs):
        """
        Creates a dataset with randomly chosen elements with or without
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
            separate_fields: If True, record fileds are saved in separate files,
                e.g. labels are stored separately from input examples.
            **kwargs: additional arguments for the Dataset initializer.
        """
        return HDDCacheDataset(self, directory, separate_fields, **kwargs)

    def info_cache_hdd(self, name_to_func, directory, **kwargs):
        """Caches the dataset on the hard disk.

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

    def collate(self, collate_func=None, func_name=None, **kwargs):
        """ Collates examples of a batch dataset or a zip dataset. """
        return CollateDataset(self, collate_func, func_name=func_name, **kwargs)

    def filter(self, func, *, func_name=None, **kwargs):
        """
        Creates a dataset containing only the elements for which `func` evaluates
        to True.
        """
        indices = np.array([i for i, d in enumerate(self) if func(d)])
        func_name = func_name or f'{_subset_hash(indices):x}'
        return SubDataset(self, indices, modifiers=f'filter({func_name})', **kwargs)

    def map(self, func, *, func_name=None, **kwargs):
        """ Creates a dataset with elements transformed with `func`. """
        return MapDataset(self, func, func_name=func_name, **kwargs)

    def map_fields(self, field_to_func, *, func_name=None, **kwargs):
        """ Creates a dataset with each element transformed with its function.

        ds.map_fields(dict(x1=func1, ..., xn=funcn)) does the same as
        ds.map(lambda r: Record(x1=func1(r.x_1), ..., xn=funcn(r.xn), x(n+1)=identity, ...))

        It is useful when using multiprocessing, which uses pickling, which
        doesn't support pickling of lambdas.
        """
        return self.map(FieldsMap(field_to_func), func_name=func_name, **kwargs)

    def permute(self, seed=53, **kwargs):
        """ Creates a permutation of the dataset. """
        indices = np.random.RandomState(seed=seed).permutation(len(self))
        return SubDataset(self, indices, modifiers="permute({seed})", **kwargs)

    def repeat(self, number_of_repeats, **kwargs):
        """
        Creates a dataset with `number_of_repeats` times the length of the
        original dataset so that every `number_of_repeats` an element is
        repeated.
        """
        return RepeatDataset(self, number_of_repeats, **kwargs)

    def split(self, *, ratio: float = None, position: int = None):
        if (ratio is None) == (position is None):
            raise ValueError("Either ratio or position needs to be specified.")
        pos = position or round(ratio * len(self))
        return self[:pos], self[pos:]

    def join(self, *other, **kwargs):
        datasets = [self] + list(other)
        info = kwargs.pop('info', {k: v for k, v in datasets[0].info.items()
                                   if all(d.info.get(k, None) == v for d in datasets[1:])})
        name = f"join(" + ",".join(x.identifier for x in datasets) + ")"
        return Dataset(name=name, info=info, data=ConcatDataset(datasets), **kwargs)

    def random(self, length=None, seed=53, **kwargs):
        """
        A modified dataset where the indexing operator returns a randomly chosen
        element which doesn't depend on the index.
        It is not clear why this would be used instead of a sampler.
        """
        return RandomDataset(self, length=length, seed=seed, **kwargs)

    def zip(self, *other, **kwargs):
        return ZipDataset([self] + list(other), **kwargs)

    def clear_hdd_cache(self):
        import inspect
        if hasattr(self, 'cache_dir'):
            shutil.rmtree(self.cache_dir)
            print(f"Deleted {self.cache_dir}")
        elif isinstance(self, MapDataset):  # lazyNormalizer
            cache_path = inspect.getclosurevars(self.func).nonlocals['f'].__self__.cache_path
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"Deleted {cache_path}")
        if isinstance(self.data, Dataset):
            self.data.clear_hdd_cache()
        elif isinstance(self.data, Sequence):
            for ds in self.data:
                if isinstance(ds, Dataset):
                    ds.clear_hdd_cache()

    @staticmethod
    def from_getitem_func(func, len, **kwargs):
        class _Data:
            def __len__(self):
                return len

            def __getitem__(self, item):
                return func(item)

        return Dataset(data=_Data(), **kwargs)

    def _print(self, *args, **kwargs):
        print(*args, f"({self.identifier})", **kwargs)


class FieldsMap:
    # TODO: use @vuf.type_checked()
    def __init__(self, field_to_func, *, mode: "Literal['override', 'replace']" = 'override'):
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


class CollateDataset(Dataset):
    def __init__(self, zip_dataset, collate_func=None, func_name=None, **kwargs):
        if isinstance(collate_func, str):
            array_type = collate_func  # to avoid lambda self-reference
            collate_func = lambda b: default_collate(b, array_type)
        self._collate = collate_func or default_collate
        batch = zip_dataset[0]
        if not all(isinstance(e, type(batch[0])) for e in batch[1:]):
            raise ValueError("All datasets must have the same element type.")
        if isinstance(zip_dataset, ZipDataset):
            info = dict(zip_dataset.data[0].info)
            for ds in zip_dataset.data[1:]:
                info.update(ds.info)
        else:
            info = zip_dataset.info
        modifier = 'collate' + ('_' + func_name if func_name else '')
        super().__init__(modifiers=modifier, info=info, data=zip_dataset, **kwargs)

    def get_example(self, idx):
        return self._collate(self.data[idx])


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
            self._cached_data = [dataset[i] for i in trange(cache_size)]

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
                    chunks = [pickle.load(f) for _ in trange(n)]
                    data = list(itertools.chain(*chunks))
            except Exception as ex:
                self._print(ex)
                self._print("Removing invalid HDD-cached dataset...")
                os.remove(cache_path)
        if data is None:
            self._print(f"Caching whole dataset in RAM...")
            data = [x for x in tqdm(dataset)]

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
                for x in tqdm(to_chunks(data)):
                    pickle.dump(x, f, protocol=4)
        super().__init__(modifiers="cache_hdd_ram", info=dataset.info, data=data, **kwargs)


class HDDCacheDataset(Dataset):
    # Caches the whole dataset on HDD
    __slots__ = ('cache_dir', 'separate_fields', 'keys')

    def __init__(self, dataset, cache_dir, separate_fields=True, consistency_check_sample_count=4,
                 **kwargs):
        modifier = 'cache_hdd' + ('_s' if separate_fields else '')
        super().__init__(modifiers=modifier, data=dataset, **kwargs)
        self.cache_dir = to_valid_path(Path(cache_dir) / self.identifier)
        self.separate_fields = separate_fields
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

    def _get_cache_path(self, idx, field=None):
        path = f"{self.cache_dir}/{idx}"
        return f"{path}_{field}.p" if field else path + '.p'

    def _get_example_or_field(self, idx, field=None):
        cache_path = self._get_cache_path(idx, field)
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
        self.verbose = verbose

    @property
    def info(self):
        if not self.initialized.value:
            self._update_info_cache()
        return self._info

    @info.setter
    def info(self, value):
        """This is called by the base initializer and (unnecessarily) by pickle
        if sharing the object among processes."""
        self._info = value

    def _get_info_cache(self):
        if self.verbose:
            print(f"{type(self).__name__}: computing/loading {self.names_str}"
                  + f" for {self.data.identifier}")
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

    def __init__(self, dataset, indices: Union[Sequence, Callable], **kwargs):
        # convert indices to smaller int type if possible
        if isinstance(indices, slice):
            self._len = len(dataset)
            start, stop, step = indices.indices(slice_len(indices, self.length))
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


class RandomDataset(Dataset):
    __slots__ = ("_rand", "_length")

    def __init__(self, dataset, length=None, seed=53, **kwargs):
        self._rand = np.random.RandomState(seed=seed)
        args = f"{seed}"
        if length is not None:
            args += f",{length}"
        super().__init__(modifiers=f"random({args})", data=dataset, **kwargs)
        self._length = length or len(dataset)

    def get_example(self, idx):
        return self.data[self._rand.randint(0, len(self.data))]

    def __len__(self):
        return self._length


class BatchDataset(Dataset):
    __slots__ = ("_length", "_batch_size", "_last_batch_size")

    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(modifiers=f"batch({batch_size})", data=dataset, **kwargs)
        self._length = len(self.data) // batch_size
        self._batch_size = batch_size
        if len(self.data) > self._length * batch_size:
            self._length += 1
        self._last_batch_size = self._batch_size - (self._length * batch_size - len(self.data))

    def get_example(self, idx):
        batch_size = self._last_batch_size if idx == self._length - 1 else self._batch_size
        i_start = idx * self._batch_size
        return tuple(self.data[i] for i in range(i_start, i_start + batch_size))

    def __len__(self):
        return self._length
