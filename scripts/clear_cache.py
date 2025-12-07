import argparse
from functools import partial

# noinspection PyUnresolvedReferences
import _context
import vidlu.data.utils as vdu
from vidlu.data import clear_hdd_cache

import dirs

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
args = parser.parse_args()

get_data = vdu.CachingDatasetFactory(
    dirs.datasets, dirs.cache,
    [partial(vdu.add_pixel_stats_to_info_lazily, cache_dir=dirs.cache)])
ds = get_data(args.ds)

clear_hdd_cache(ds)
