import typing as T
from numbers import Real

import numpy as np
import torch.utils.data as tud

from vidlu.data.data_loader import ZipDataLoader, DataLoader
from vidlu.data.dataset import Dataset
from vidlu.data.record import Record
import vidlu.data.utils.samplers as samplers
from vidlu.utils.misc import broadcast


def _1_or_more(type):
    return T.Union[type, T.Sequence[type]]


class TDataLoaderF(T.Protocol):
    def __call__(self, dataset: T.Sequence, **kwargs) -> T.Iterable: ...


class TMultiDataLoaderF(T.Protocol):
    def __call__(self, datasets: _1_or_more(T.Sequence), data_loader_f: _1_or_more(TDataLoaderF),
                 **kwargs) -> T.Iterable: ...


def zip_data_loader(*datasets: T.Sequence,
                    data_loader_f: TDataLoaderF,
                    primary_index: T.Optional[T.Union[int, T.Literal['equal']]] = 0,
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
    dataset_count = len(datasets)
    data_loader_fs = broadcast(data_loader_f, dataset_count, seq_type=list)
    kwargs = {k: broadcast(v, dataset_count, seq_type=list) for k, v in kwargs.items()}
    data_loaders = [dl_f(ds, **{k: kwargs[k][i] for k in kwargs})
                    for i, (ds, dl_f) in enumerate(zip(datasets, data_loader_fs))]
    return ZipDataLoader(*data_loaders, primary_index=primary_index)


def simple_or_zip_data_loader(*datasets,
                              data_loader_f: TDataLoaderF,
                              primary_index: T.Optional[T.Union[int, T.Literal['equal']]] = 0,
                              **kwargs):
    return data_loader_f(*datasets, **kwargs) if len(datasets) == 1 else \
        zip_data_loader(*datasets, data_loader_f=data_loader_f, primary_index=primary_index,
                        **kwargs)


def simple_or_multi_data_loader(*datasets,
                                data_loader_f: TDataLoaderF,
                                multi_data_loader_f: TMultiDataLoaderF,
                                **kwargs):
    return data_loader_f(*datasets, **kwargs) if len(datasets) == 1 else \
        multi_data_loader_f(*datasets, data_loader_f=data_loader_f, **kwargs)


def mixed_data_loader(*datasets: T.Sequence[Dataset],
                      data_loader_f: TDataLoaderF,
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


# def semisup_data_loader(
#         ds_l: Dataset, ds_u: Dataset,
#         data_loader_f: TDataLoaderF,
#         labeled_incidence: T.Union[Real, T.Callable[[int, int], Real]] = lambda l, u: max(1, l / (l + u)),
#         **kwargs):
#     if callable(labeled_incidence):
#         labeled_incidence = labeled_incidence(len(ds_l), len(ds_u))
#
#     return data_loader_f()


def morsic_semisup_data_loader(
        ds_l: Dataset, ds_u: Dataset,
        data_loader_f: TDataLoaderF,
        labeled_multiplier: T.Union[int, T.Callable[[int, int], int]] = \
                lambda l, u: max(1, int(u / l)),
        **kwargs):
    nl, nu = len(ds_l), len(ds_u)
    if callable(labeled_multiplier):
        labeled_multiplier = labeled_multiplier(nl, nu)
    ds_all = ds_l.join(ds_u)
    indices_l = list(range(nl)) * labeled_multiplier
    indices_u = list(range(nl, nl + nu))
    indices = indices_l * labeled_multiplier + indices_u
    sampler = tud.SubsetRandomSampler(indices=indices)
    if kwargs.get("shuffle", False):
        raise ValueError("The shuffle argument should be False.")
    return data_loader_f(ds_all, sampler=sampler, shuffle=False, **kwargs)


# def multiset_data_loader_2(
#         ds_l: Dataset, ds_u: Dataset,
#         data_loader_f: TDataLoaderF,
#         multiplier: float,
#         example_weights: T.Sequence[int],
#         **kwargs):
#     nl, nu = len(ds_l), len(ds_u)
#     example_weights = [np.ones(nl), np.ones(nu)] if example_weights is None \
#         else list(map(np.array, example_weights))
#     example_weights = np.concatenate(example_weights, axis=0)
#     indices = [i for i, w in enumerate(example_weights * multiplier)
#                for _ in range(int(w + 0.5))]
#     ds_all = ds_l.join(ds_u)
#     sampler = tud.SubsetRandomSampler(indices=indices)
#     return data_loader_f(ds_all, sampler=sampler, **kwargs)


def multiset_data_loader(
        dataset: T.Tuple[Dataset], data_loader_f: TDataLoaderF,
        multiplicities: T.Sequence[T.Sequence[int]] = None,
        **kwargs):
    sampler = samplers.multiset_sampler(multiplicities, dataset)
    return data_loader_f(dataset, sampler=sampler, **kwargs)
