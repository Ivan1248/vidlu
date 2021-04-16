import shutil
from argparse import Namespace
from vidlu.utils.func import partial
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

from vidlu.data import DatasetFactory
from vidlu.data.record import Record
from vidlu.utils import path


# Standardization ##################################################################################

def compute_pixel_mean_std(dataset, scale01=False, progress_bar=False):
    pbar = tqdm if progress_bar else lambda x: x
    mvn = np.array([(x.mean((0, 1)), x.var((0, 1)), np.prod(x.shape[:2]))
                    for x in pbar(dataset.map(lambda r: np.array(r.x)))])
    means, vars_, ns = [mvn[:, i] for i in range(3)]  # means, variances, pixel counts
    ws = ns / ns.sum()  # image weights (pixels in image / pixels in all images)
    mean = ws.dot(means)  # mean pixel
    var = vars_.mean(0) + ws.dot(means ** 2) - mean ** 2  # pixel variance
    std = np.sqrt(var)  # pixel standard deviation
    return (mean / 255, std / 255) if scale01 else (mean, std)


# Image statistics cache ###########################################################################

# not local for picklability, used only in add_image_statistics_to_info_lazily
def _get_standardization(_, ds_with_info):
    return ds_with_info.info.cache['standardization']


# not local for picklability, used only in add_image_statistics_to_info_lazily
def _compute_pixel_mean_std_d(ds):
    mean, std = compute_pixel_mean_std(ds, scale01=True, progress_bar=True)
    return Namespace(mean=mean, std=std)


def _map_record(func, r):
    def map_(k):
        return lambda: func(r[k])

    return Record({f'{k}_': map_(k) for k in r.keys()})


def add_image_statistics_to_info_lazily(parted_dataset, cache_dir):
    pds = parted_dataset
    try:
        stats_ds = pds.trainval if 'trainval' in pds else pds.train.join(pds.val)
    except KeyError:
        part_name, stats_ds = next(iter(pds.items()))
        warnings.warn('The parted dataset object has no "trainval" or "train" and "val" parts.'
                      + f' "{part_name}" is used instead.')
    ds_with_info = stats_ds.info_cache_hdd(
        dict(standardization=_compute_pixel_mean_std_d), Path(cache_dir) / 'dataset_statistics')

    def cache_transform(ds):
        return ds.info_cache(
            dict(standardization=partial(_get_standardization, ds_with_info=ds_with_info)))

    return _map_record(cache_transform, pds)


# Caching ##########################################################################################

def cache_data_lazily(parted_dataset, cache_dir, min_free_space=20 * 1024 ** 3):
    def transform(ds):
        elem_size = ds.example_size(sample_count=4)
        size = len(ds) * elem_size
        free_space = shutil.disk_usage(cache_dir).free
        space_left = free_space - size

        ds_cached = ds.cache_hdd(f"{cache_dir}/datasets")
        has_been_cached = path.get_size(ds_cached.cache_dir) > size * 0.1
        if has_been_cached or space_left >= min_free_space:
            ds = ds_cached
        else:
            ds_cached.delete_cache()
            del ds_cached
            warnings.warn(f'The dataset {ds.identifier} will not be cached because there is not'
                          + f' much space left.'
                          + f'\nAvailable space: {free_space / 2 ** 30:.3f} GiB.'
                          + f'\nData size: {elem_size * len(ds) / 2 ** 30:.3f} GiB (subset),'
                          + f' {size / 1024 ** 3:.3f} GiB (all).')
        return ds

    return _map_record(transform, parted_dataset)


class CachingDatasetFactory(DatasetFactory):
    def __init__(self, datasets_dir_or_factory, cache_dir, parted_ds_transforms=()):
        ddof = datasets_dir_or_factory
        super().__init__(ddof.datasets_dirs if isinstance(ddof, DatasetFactory) else ddof)
        self.cache_dir = cache_dir
        self.parted_ds_transforms = parted_ds_transforms

    def __call__(self, ds_name, **kwargs):
        pds = super().__call__(ds_name, **kwargs)
        for transform in self.parted_ds_transforms:
            pds = transform(pds)
        return cache_data_lazily(pds, self.cache_dir)
