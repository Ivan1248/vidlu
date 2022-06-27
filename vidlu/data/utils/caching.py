import shutil
from argparse import Namespace
from vidlu.utils.func import partial
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

from vidlu.data import DatasetFactory
from vidlu.data.record import Record, LazyField
from vidlu.utils import path
from vidlu.data.utils import class_incidence
from vidlu.data.data_loader import SingleDataLoader


# Standardization ##################################################################################

def compute_pixel_stats(ds, div255=False, progress_bar=False, num_workers=4):
    pbar = partial(tqdm, desc='compute_pixel_stats') if progress_bar else lambda x: x
    images = (np.array(r.image) for r in pbar(SingleDataLoader(ds, num_workers=num_workers)))
    mvn = np.array([(x.mean((0, 1)), x.var((0, 1)), np.prod(x.shape[:2])) for x in images])
    means, vars_, ns = [mvn[:, i] for i in range(3)]  # means, variances, pixel counts
    ws = ns / ns.sum()  # image weights (pixels in image / pixels in all images)
    mean = ws.dot(means)  # mean pixel
    var = vars_.mean(0) + ws.dot(means ** 2) - mean ** 2  # pixel variance
    std = np.sqrt(var)  # pixel standard deviation
    return (mean / 255, std / 255) if div255 else (mean, std)


# Pixel statistics cache ###########################################################################


# not local for picklability, used only in add_image_statistics_to_info_lazily
def _compute_pixel_stats_d(ds):
    mean, std = compute_pixel_stats(ds, div255=True, progress_bar=True)
    return Namespace(mean=mean, std=std)


def add_pixel_stats_to_info_lazily(ds, cache_dir):
    return ds.info_cache_hdd(dict(pixel_stats=_compute_pixel_stats_d), cache_dir,
                             simplify_dataset=lambda ds: ds[:4], name='pixel_stats')


def add_segmentation_class_info_lazily(ds, cache_dir):
    if 'seg_map' not in ds[0].keys():
        return ds

    def add_seg_class_info(example):
        info = LazyField(partial(class_incidence.example_seg_class_info, example))
        return Record(classes_=lambda: info()['classes'],
                      class_incidences_=lambda: info()['class_incidences'],
                      class_aabbs_=lambda: info()['class_aabbs'])

    return ds.map(lambda r: r.join(add_seg_class_info(r)))
    # ds = ds.info_cache_hdd(dict(seg_class_info=seg_class_info), cache_dir, recompute=False,
    #                        simplify_dataset=lambda ds: ds[:4])
    # return ds.enumerate().map_unpack(
    #     lambda i, r: Record(
    #         r, class_seg_boxes_=lambda: ds.info.seg_class_info['class_segment_boxes'][i]))


# Caching ##########################################################################################

def cache_data_lazily(ds, cache_dir, min_free_space=20 * 2 ** 30):
    ds_cached = ds.cache_hdd(f"{cache_dir}/datasets")

    elem_size = ds.example_size(sample_count=4)
    size = len(ds) * elem_size
    free_space = shutil.disk_usage(cache_dir).free
    cached_size = path.get_size(ds_cached.cache_dir)

    if cached_size > size * 0.1 or free_space + cached_size - size >= min_free_space:
        return ds_cached
    else:
        warnings.warn(f'The dataset {ds.identifier} will not be cached because there is not'
                      + f' much space left.'
                      + f' Available space: {(free_space + cached_size) / 2 ** 30:.3f} GiB.'
                      + f' Data size: {size / 2 ** 30:.3f} GiB.')
        ds_cached.delete_cache()
        return ds


class CachingDatasetFactory(DatasetFactory):  # TODO: delete
    def __init__(self, datasets_dir_or_factory, cache_dir, transforms=()):
        ddof = datasets_dir_or_factory
        super().__init__(ddof.datasets_dirs if isinstance(ddof, DatasetFactory) else ddof)
        self.transforms = transforms
        self.cache_dir = cache_dir

    def __call__(self, ds_name, **kwargs):
        ds = super().__call__(ds_name, **kwargs)
        for transform in self.transforms:
            ds = transform(ds)
        return cache_data_lazily(ds, self.cache_dir)
