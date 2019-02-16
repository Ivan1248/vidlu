import os
import shutil
import multiprocessing
from tqdm import tqdm
import pickle
from collections import Sequence
import inspect
import warnings

import numpy as np
import torch

from vidlu.data import Dataset, Record, serialized_sizeof
from vidlu import data
from vidlu.problem import dataset_to_problem, Problem
from vidlu.data_processing.image.random_transforms import random_fliplr_with_label, augment_cifar
from vidlu.utils import path

from torchvision.transforms.functional import to_tensor


# Normalization ####################################################################################


class LazyNormalizer:
    def __init__(self, ds, cache_dir, max_sample_size=10000):
        self.ds = ds
        if len(self.ds) > max_sample_size:
            self.ds = self.ds.permute()[:max_sample_size]

        self.initialized = multiprocessing.Value('i', 0)
        self.mean, self.std = (multiprocessing.Array('f', x) for x in
                               self.get_input_mean_std([ds[0], ds[1]]))

        self.cache_dir = f"{cache_dir}/lazy-normalizer-cache"
        self.cache_path = f"{self.cache_dir}/{ds.name}.p"

    @staticmethod
    def get_input_mean_std(dataset):
        ms = np.array([(x.mean((0, 1)), x.std((0, 1))) for x, y in dataset])
        m, s = ms.mean(0)
        return m, s

    def _initialize(self):
        mean_std = None
        if os.path.exists(self.cache_path):
            try:
                print(f"Loading dataset statistics for {self.ds.name}")
                with open(self.cache_path, 'rb') as cache_file:
                    mean_std = pickle.load(cache_file)
            except:
                os.remove(self.cache_path)
        if mean_std is None:
            print(f"Computing dataset statistics for {self.ds.name}")
            mean_std = self.get_input_mean_std(tqdm(self.ds))
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(f"{self.cache_path}", 'wb') as cache_file:
                pickle.dump(mean_std, cache_file, protocol=4)
        self.mean[:], self.std[:] = mean_std

    def normalize(self, x):
        with self.initialized.get_lock():
            if not self.initialized.value:  # lazy
                self._initialize()
                print(f'mean = {np.array(self.mean)} std = {np.array(self.std)}')
                self.initialized.value = True
                # CAUTION! np.array(self.mean) != np.array(self.mean.value)
        return ((x - self.mean) / self.std).astype(np.float32)


# Cached dataset with normalized inputs ############################################################

def cache_data_and_normalize_inputs(parted_dataset, cache_dir):
    normalizer = LazyNormalizer(parted_dataset.trainval, cache_dir)
    elem_size = serialized_sizeof(parted_dataset.trainval[0])
    size = elem_size * len(parted_dataset.all)
    free_space = shutil.disk_usage(cache_dir).free
    dataset_space_proportion = size / free_space

    def transform(ds):
        ds = ds.map(lambda r: Record(x=normalizer.normalize(r.x), y=r.y), func_name='normalize')
        ds_cached = ds.cache_hdd(f"{cache_dir}/datasets")
        is_already_cached = path.get_size(ds_cached.cache_dir) > size * 0.05
        if is_already_cached or dataset_space_proportion < 0.5:
            ds = ds_cached
        else:
            ds_cached.delete_cache()
            del ds_cached
            warnings.warn(f'The dataset {ds.full_name} will not be cached because it is too large.'
                          + f' Available space: {free_space / 2 ** 30} GiB.'
                          + f' Data size (subset / all): {elem_size * len(ds)} GiB / {size} GiB.')
        return ds

    return parted_dataset.with_transform(transform)


def clear_dataset_hdd_cache(ds):
    if hasattr(ds, 'cache_dir'):
        shutil.rmtree(ds.cache_dir)
        print(f"Deleted {ds.cache_dir}")
    elif isinstance(ds, data.dataset.MapDataset):  # lazyNormalizer
        cache_path = inspect.getclosurevars(ds.func).nonlocals['f'].__self__.cache_path
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Deleted {cache_path}")
    for k in dir(ds):
        a = getattr(ds, k)
        if isinstance(a, Dataset):
            clear_dataset_hdd_cache(a)
        elif isinstance(a, Sequence):
            for b in a:
                if isinstance(b, Dataset):
                    clear_dataset_hdd_cache(b)


# Augmentation #####################################################################################

def get_default_augmentation_func(dataset):
    if any(lambda x: dataset.name.lower().startswith(x)
           for x in ['cifar', 'cifar100', 'tinyimagenet']):
        return lambda xy: (augment_cifar(xy[0]), xy[1])
    elif dataset.info['problem_id'] == 'semseg':
        return random_fliplr_with_label
    else:
        return lambda x: x


# Torch ############################################################################################

# TODO
def to_torch(dataset, device):
    problem = dataset_to_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        breakpoint()
        transform = lambda r: Record(x=torch.from_numpy(r.x), y=torch.from_numpy(r.y))
    else:
        raise NotImplementedError()
    return dataset.map(transform)
