import argparse

from _context import vidlu

from vidlu.data import DatasetFactory
from vidlu.data.misc import serialized_sizeof
from vidlu import data_utils
from vidlu.utils.presentation.visualization import view_predictions
from vidlu.utils.tree import print_tree

import dirs

# python view_dataset.py
#   inaturalist2018 train
#   voc2012 test
#   wilddash bench

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--permute', action='store_true')
args = parser.parse_args()

pds = DatasetFactory(dirs.DATASETS)(args.ds)
pds = data_utils.cache_data_and_normalize_inputs(pds, dirs.CACHE)
ds = pds[args.part]

print("Name:", ds.name)
print("Info:")
print_tree(ds.info, depth=1)
print("Number of examples:", len(ds))
print(f"Size estimate: {serialized_sizeof(ds[0]) * len(ds) / 2 ** 30:.3f} GiB")

if 'class_count' not in ds.info:
    ds.info['class_count'] = 2

if args.augment:
    ds = ds.map(data_utils.get_default_augmentation_func(ds))

if args.permute:
    ds = ds.permute()

view_predictions(ds, infer=lambda x: ds[0][1])
