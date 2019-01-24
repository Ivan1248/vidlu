import argparse

from _context import vidlu

from vidlu import data_utils, dirs

# python clear_cache.py <dsid>

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

dss = data_utils.get_cached_dataset_set_with_normalized_inputs(args.ds)

for k, ds in dss.items():
    data_utils.clear_dataset_hdd_cache(ds)
