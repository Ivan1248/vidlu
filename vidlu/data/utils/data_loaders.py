import typing as T
import os
from functools import partial

import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, Dataset, Sampler

from vidlu.data import DataLoader, ZipDataLoader, CombinedDataLoader
from vidlu.data.record import Record
import vidlu.data.utils.samplers as samplers
import vidlu.utils.distributed as vud
from vidlu.utils.misc import broadcast
from vidlu.utils.func import params


# Annotation types

def _1_or_more(type):
    return T.Union[type, T.Sequence[type]]


class TDataLoaderF(T.Protocol):
    def __call__(self, dataset: T.Sequence, **kwargs) -> T.Iterable: ...


class TMultiDataLoaderF(T.Protocol):
    def __call__(self, datasets: _1_or_more(T.Sequence), data_loader_f: _1_or_more(TDataLoaderF),
                 **kwargs) -> T.Iterable: ...


# Factories

def make_data_loaders(data_loader_f, datasets, kwargs):
    dataset_count = len(datasets)
    data_loader_fs = broadcast(data_loader_f, dataset_count, seq_type=list)
    kwargs = {k: broadcast(v, dataset_count, seq_type=list) for k, v in kwargs.items()}
    data_loaders = [dl_f(ds, **{k: kwargs[k][i] for k in kwargs})
                    for i, (ds, dl_f) in enumerate(zip(datasets, data_loader_fs))]
    return data_loaders


def combined_data_loader(*datasets, data_loader_f, collate_fn=None, primary_index='shortest',
                         **kwargs):
    data_loaders = make_data_loaders(data_loader_f, datasets, kwargs)
    if not all(dl.collate_fn for dl in data_loaders):
        raise ValueError("data_loader_f should have collate_fn=None.")
    return CombinedDataLoader(*data_loaders, collate_fn=collate_fn, primary_index=primary_index)


def zip_data_loader(*datasets: T.Sequence,
                    data_loader_f: TDataLoaderF = DataLoader,
                    primary_index: T.Optional[
                        T.Union[int, T.Literal['longest', 'shortest', 'equal']]] = 0,
                    **kwargs):
    """Creates a ZipDataLoader instance with `D = len(datasets)` datasets.

    Args:
        datasets: A sequence of `D` datasets.
        data_loader_f: 1 or a sequence of `D` data loader factories.
        primary_index: Index of the dataset that all examples must be sampled
            from in an epoch.
        **kwargs: Data loader (factory) arguments or length `D` sequences of
            arguments. Example: `batch_size=[32, 512], num_workers=2`.

    Returns:
        A ZipDataLoader instance yielding around
        `len(datasets[primary_ds_index]) / batch_size` batches (depending on
        the value of `drop_last)`.
    """
    data_loaders = make_data_loaders(data_loader_f, datasets, kwargs)
    if not all(dl.drop_last for dl in data_loaders):
        raise ValueError("data_loader_f should have drop_last=True.")
    return ZipDataLoader(*data_loaders, primary_index=primary_index)


def auto_data_loader(*datasets,
                     dl_f: TDataLoaderF = DataLoader,
                     multi_dl_f: T.Union[TMultiDataLoaderF, T.Literal['zip', 'combine']] = 'zip',
                     primary_index: T.Optional[T.Union[int, T.Literal['equal']]] = 0,
                     **kwargs):
    if isinstance(multi_dl_f, str):
        multi_dl_f = zip_data_loader if multi_dl_f == 'zip' else combined_data_loader
    return dl_f(*datasets, **kwargs) if len(datasets) == 1 else \
        multi_dl_f(*datasets, data_loader_f=dl_f, primary_index=primary_index,
                   **kwargs)


def mixed_data_loader(*datasets: T.Sequence[Dataset],
                      data_loader_f: TDataLoaderF = DataLoader,
                      dataset_weights: T.Sequence[int] = None,
                      example_weights: T.Sequence[T.Sequence[float]] = None,
                      **kwargs):
    N = sum(len(ds) for ds in datasets)
    if dataset_weights is None:
        dataset_weights = [len(ds) / N for ds in datasets]  # proportional to dataset size
    else:
        if len(dataset_weights) != len(datasets):
            raise RuntimeError(
                "The number of dataset weights does not match the number of datasets.")
        dataset_weights = np.array(dataset_weights) / sum(dataset_weights)
    if example_weights is not None and any(
            len(ew) != len(ds) for ew, ds in zip(example_weights, datasets)):
        raise RuntimeError(
            "Lengths of vectors of example weights do not match lengths of datasets.")

    datasets = [ds.map(lambda r: Record(r, ds_index=i)) for i, ds in enumerate(datasets)]
    ds_all = datasets[0].join(*datasets[1:])
    if example_weights is not None:
        example_weights = list(map(np.array, example_weights))
        weights = sum(
            [list(wd / we.sum() * we) for wd, we in zip(dataset_weights, example_weights)], [])
    else:
        weights = sum([[wd / len(ds)] * len(ds) for ds, wd in zip(datasets, dataset_weights)], [])
    sampler = tud.WeightedRandomSampler(weights=weights,
                                        num_samples=sum(len(ds) for ds in datasets),
                                        replacement=False)
    return data_loader_f(ds_all, sampler=sampler, **kwargs)


def mixed_semisup_collate(batch, collate):
    labeled = collate([r.labeled for r in batch])
    y_l = collate([r.y for lab, r in zip(labeled, batch) if lab])
    x = collate([r.x for lab, r in zip(labeled, batch) if lab]
                + [r.x for lab, r in zip(labeled, batch) if not lab])
    return Record(x_l=x[:len(y_l)], x_u=x[len(y_l):], y_l=y_l)


def morsic_semisup_data_loader(
        ds_l: Dataset, ds_u: Dataset,
        data_loader_f: TDataLoaderF = DataLoader,
        labeled_multiplier: T.Union[int, T.Callable[[int, int], int]] = \
                lambda l, u: max(1, int(u / l)),
        **kwargs):
    if kwargs.get("shuffle", False):
        raise ValueError("The shuffle argument should be False.")
    nl, nu = len(ds_l), len(ds_u)
    if callable(labeled_multiplier):
        labeled_multiplier = labeled_multiplier(nl, nu)
    ds_l = ds_l.map(lambda r: Record(x=r[0], y=r[1], labeled=True))
    ds_u = ds_u.map(lambda r: Record(x=r[0], y=None, labeled=False))
    ds_all = ds_l.join(ds_u)
    indices_l = list(range(nl)) * labeled_multiplier
    indices_u = list(range(nl, nl + nu))
    indices = indices_l * labeled_multiplier + indices_u
    sampler = tud.SubsetRandomSampler(indices=indices)
    kwargs['collate_fn'] = partial(
        mixed_semisup_collate,
        collate=(kwargs if 'collate_fn' in kwargs else params(data_loader_f))['collate_fn'])
    return data_loader_f(ds_all, sampler=sampler, shuffle=False, **kwargs)


def multiset_data_loader(
        dataset: T.Tuple[Dataset],
        data_loader_f: TDataLoaderF = DataLoader,
        multiplicities: T.Sequence[T.Sequence[int]] = None,
        **kwargs):
    sampler = samplers.multiset_sampler(multiplicities, dataset)
    return data_loader_f(dataset, sampler=sampler, **kwargs)


# Utilities for composite data loaders

def is_dataloader(obj):
    return 'DataLoader' in type(obj).__name__


def get_children(data_loader, filter_=is_dataloader):
    for k, ch in vars(data_loader):
        if filter_(ch):
            yield ch
        elif isinstance(ch, T.Sequence) and len(ch) > 0 and filter_(ch[0]):
            yield from ch


def get_components(data_loader, filter_=is_dataloader):
    for dl in get_children(data_loader, filter_=filter_):
        yield from get_components(dl, filter_=filter_)
    yield data_loader


def update_components(data_loader, func, filter_=is_dataloader):
    for k, ch in vars(data_loader):
        if is_dataloader(ch):
            setattr(data_loader, k, update_components(ch, func, filter_=filter_))
        elif isinstance(ch, T.Sequence) and len(ch) > 0 and filter_(ch[0]):
            setattr(data_loader, k, [update_components(c, func, filter_=filter_) for c in ch])
    return data_loader


def is_sampler(obj):
    return 'Sampler' in type(obj).__name__


def get_samplers(data_loader, filter_=is_sampler):
    for dl in get_components(data_loader):
        yield from get_children(dl, filter_=filter_)


# Distributed training


class _DistributedDataLoaderWrapper:
    def __init__(self, data_loader: DataLoader) -> None:
        """A wrapper for `torch.utils.data.DataLoader` that automatically calls `set_epoch` for
        changing the ordering in different epochs of distributed training.

        Calling `set_epoch` before creating the iterator is necessary for the ordering to change
        between epochs. More here:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

        Args:
            data_loader: The wrapped DataLoader.
        """
        self.__dict__.update(data_loader.__dict__)
        self.data_loader = data_loader
        self._pass_index = 0

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for sampler in get_samplers(self.data_loader):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._pass_index)
        self._pass_index += 1
        return iter(self.data_loader)


def _get_distributed_sampler(data_loader: DataLoader, **kwargs) -> DistributedSampler:
    kwargs = {
        **dict(shuffle=isinstance(data_loader.sampler, RandomSampler), seed=torch.initial_seed()),
        **kwargs}

    if isinstance(data_loader.sampler, (RandomSampler, SequentialSampler)):
        return DistributedSampler(data_loader.dataset, **kwargs)
    from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
    return DistributedSamplerWrapper(data_loader.sampler, **kwargs)


def make_data_loader_distributed(data_loader, keep_worker_init_fn=False,
                                 **distributed_sampler_kwargs):
    # https://github.com/Lightning-AI/lightning/blob/deb70464967f4a996714874ebf44861c3481e0ff/src/lightning/fabric/fabric.py#L310
    """Set up a data_loader for distributed training with different samplers and RNG initializations.

    Based on lightning.Fabric._setup_dataloader.

    Args:
        data_loader: The data_loader to process.

    Returns:
        The wrapped data_loader.
    """

    def update(dl):
        if not hasattr(dl, 'sampler'):
            return dl

        # re-instantiates the data_loader with the updated sampler
        from lightning.fabric.utilities.data import _update_dataloader
        sampler = _get_distributed_sampler(dl, **distributed_sampler_kwargs)
        dl = _update_dataloader(dl, sampler)

        if not keep_worker_init_fn:
            # adds worker_init_fn for correct seeding in worker processes
            # TODO: Update vidlu.data.data_loader.worker_init_fn?
            if hasattr(data_loader, "worker_init_fn"):
                from lightning.fabric.utilities.seed import pl_worker_init_function
                data_loader.worker_init_fn = partial(pl_worker_init_function,
                                                     rank=vud.get_global_rank())

        return dl

    data_loader = update_components(data_loader, update)
    return _DistributedDataLoaderWrapper(data_loader)
