import argparse

from scripts import dirs
from vidlu.data.datasets import DatasetFactory
from vidlu.data_utils import cache_data_and_normalize_inputs, clear_dataset_hdd_cache

# python clear_cache.py <dsid>

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

get_data = DatasetFactory(dirs)
pds = cache_data_and_normalize_inputs(get_data(args.ds))

for k, ds in pds.items():
    clear_dataset_hdd_cache(ds)
