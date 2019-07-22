import os
import shutil
import multiprocessing
from argparse import Namespace
from functools import partialmethod
from pathlib import Path

import torch
from tqdm import tqdm
import warnings

import numpy as np

from vidlu.data import DatasetFactory, default_collate
from vidlu.utils import path
from vidlu.utils.collections import NameDict


# Standardization ##################################################################################

def compute_pixel_mean_std(dataset, progress_bar=False):
    pbar = tqdm if progress_bar else lambda x: x
    mvn = np.array([(x.mean((0, 1)), x.var((0, 1)), np.prod(x.shape[:2]))
                    for x in pbar(dataset.map(lambda r: np.array(r.x)))])
    means, vars, ns = [mvn[:, i] for i in range(3)]  # means, variances, pixel counts
    ws = ns / ns.sum()  # image weights (pixels in image / pixels in all images)
    mean = ws.dot(means)  # mean pixel
    var = vars.mean(0) + ws.dot(means ** 2) - mean ** 2  # pixel variance
    return mean, np.sqrt(var)  # pixel mean, pixel standard deviation


# Cached dataset with normalized inputs ############################################################

def add_image_statistics_to_info_lazily(parted_dataset, cache_dir):
    def compute_pixel_mean_std_d(ds):
        return Namespace(**dict(zip(['mean', 'std'], compute_pixel_mean_std(ds, True))))
    ds_with_info = parted_dataset.trainval.info_cache_hdd(
        dict(standardization=compute_pixel_mean_std_d), Path(cache_dir) / 'dataset_statistics')
    return parted_dataset.with_transform(
        lambda ds: ds.info_cache(
            dict(standardization=lambda ds: ds_with_info.info.cache['standardization'])))

    """
    imstat = LazyImageStatisticsComputer(parted_dataset.trainval, cache_dir)

    def transform(ds):
        ds.info.cache.standardization = NameDict(**imstat.stats, standardized=False)
        return ds

    return parted_dataset.with_transform(transform)
    """


def cache_data_lazily(parted_dataset, cache_dir):
    elem_size = parted_dataset.trainval.approx_example_sizeof()
    size = elem_size
    free_space = shutil.disk_usage(cache_dir).free
    dataset_space_proportion = size / free_space

    def transform(ds):
        ds_cached = ds.cache_hdd(f"{cache_dir}/datasets")
        has_been_cached = path.get_size(ds_cached.cache_dir) > size * 0.1
        if has_been_cached or dataset_space_proportion < 0.5:
            ds = ds_cached
        else:
            ds_cached.delete_cache()
            del ds_cached
            warnings.warn(f'The dataset {ds.identifier} will not be cached because it is too large.'
                          + f' Available space: {free_space / 2 ** 30} GiB.'
                          + f' Data size: {elem_size * len(ds)} GiB (subset), {size} GiB (all).')
        return ds

    return parted_dataset.with_transform(transform)


class CachingDatasetFactory(DatasetFactory):
    def __init__(self, datasets_dir_or_factory, cache_dir, add_statistics=False):
        ddof = datasets_dir_or_factory
        super().__init__(ddof.datasets_dir if isinstance(ddof, DatasetFactory) else ddof)
        self.add_statistics = add_statistics
        self.cache_dir = cache_dir

    def __call__(self, ds_name, **kwargs):
        pds = super().__call__(ds_name, **kwargs)
        if self.add_statistics:
            pds = add_image_statistics_to_info_lazily(pds, self.cache_dir)
        return cache_data_lazily(pds, self.cache_dir)


# DataLoader class with collate function that supports Record examples

class DataLoader(torch.utils.data.DataLoader):
    __init__ = partialmethod(torch.utils.data.DataLoader.__init__, shuffle=True,
                             collate_fn=default_collate, drop_last=True)
