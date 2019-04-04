import argparse

from _context import vidlu
from vidlu.data_utils import CachingDatasetFactory

import dirs

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

get_data = CachingDatasetFactory(dirs.DATASETS, cache_dir=dirs.CACHE)
pds = get_data(args.ds)

for k, ds in pds.items():
    ds.clear_hdd_cache()
