import os
import shutil
import multiprocessing
from tqdm import tqdm
import warnings

import numpy as np

from vidlu.data import DatasetFactory
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


class LazyImageStatisticsComputer:
    def __init__(self, dataset, cache_dir, max_sample_size=10000):
        self.dataset = dataset
        if len(self.dataset) > max_sample_size:
            self.dataset = self.dataset.permute()[:max_sample_size]
        self.initialized = multiprocessing.Value('i', 0)
        self.single_channel = len(np.array(dataset[0].x).shape) == 2
        value_type = multiprocessing.Value if self.single_channel else multiprocessing.Array
        self._mean, self._std = (value_type('f', x) for x in compute_pixel_mean_std(dataset[:2]))
        self.cache_dir = f"{cache_dir}/dataset-statistics"
        self.cache_path = f"{self.cache_dir}/{dataset.identifier}.txt"

    @property
    def stats(self):
        self._initialize_if_necessary()
        mean, std = ((self._mean.item, self._std.item) if self.single_channel
                     else (np.array(self._mean), np.array(self._std)))
        return NameDict(mean=mean, std=std)

    def _initialize(self):
        mean_std = None
        if os.path.exists(self.cache_path):
            try:
                mean_std = np.loadtxt(self.cache_path)
            except:
                os.remove(self.cache_path)
        if mean_std is None:
            print(f"Computing dataset statistics for {self.dataset.name}")
            mean_std = compute_pixel_mean_std(self.dataset, progress_bar=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savetxt(self.cache_path, mean_std, fmt='%12.8f')
        if self.single_channel:
            self._mean.item, self._std.item = mean_std
        else:
            self._mean[:], self._std[:] = mean_std

    def _initialize_if_necessary(self):
        with self.initialized.get_lock():
            if not self.initialized.value:  # lazy
                self._initialize()
                self.initialized.value = True


# Cached dataset with normalized inputs ############################################################

def add_image_statistics_to_info_lazily(parted_dataset, cache_dir):
    imstat = LazyImageStatisticsComputer(parted_dataset.trainval, cache_dir)

    def transform(ds):
        ds.info.standardization = NameDict(**imstat.stats, standardized=False)
        return ds

    return parted_dataset.with_transform(transform)


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
    def __init__(self, datasets_dir, cache_dir):
        super().__init__(datasets_dir)
        self.datasets_dir = datasets_dir
        self.cache_dir = cache_dir

    def __call__(self, ds_name, **kwargs):
        pds = super().__call__(ds_name, **kwargs)
        pds = add_image_statistics_to_info_lazily(pds, self.cache_dir)
        return cache_data_lazily(pds, self.cache_dir)


"""
# Torch ############################################################################################

# TODO
def to_torch(dataset, device):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        transform = lambda r: Record(x=torch.from_numpy(r.x), y=torch.from_numpy(r.y))
    else:
        raise NotImplementedError()
    return dataset.map(transform)



def dataset_to_torch(ds):
    from torchvision.transforms.functional import to_tensor
    def label_to_tensor(x):
        if x.dtype == np.int8:  # PyTorch doesn't support np.int8 -> torch.int8
            return torch.tensor(x.astype(np.uint8), dtype=torch.int8)
        return torch.from_numpy(x)

    return ds.map(lambda r: Record(x=to_tensor(r.x), y=label_to_tensor(r.y)))
"""
