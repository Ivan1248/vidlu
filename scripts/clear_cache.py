import argparse
from functools import partial

# noinspection PyUnresolvedReferences
import _context
import vidlu.data.utils as vdu

import dirs

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

get_data = vdu.CachingDatasetFactory(
    dirs.DATASETS, dirs.CACHE,
    [partial(vdu.add_image_statistics_to_info_lazily, cache_dir=dirs.CACHE)])
pds = get_data(args.ds)

for k, ds in pds.items():
    ds.clear_hdd_cache()
