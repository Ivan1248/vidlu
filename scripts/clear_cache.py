import argparse

from _context import vidlu
from vidlu.data.datasets import DatasetFactory
from vidlu.data_utils import cache_data_and_normalize_inputs, clear_dataset_hdd_cache

import dirs

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

get_data = DatasetFactory(dirs.DATASETS)
pds = cache_data_and_normalize_inputs(get_data(args.ds), cache_dir=dirs.CACHE)

for k, ds in pds.items():
    ds.clear_hdd_cache()
