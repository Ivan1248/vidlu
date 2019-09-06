import argparse

from _context import vidlu
from vidlu.data.misc import pickle_sizeof
from vidlu.transforms import image, jitter
from vidlu.transforms.image import to_pil
from vidlu.utils.presentation.visualization import view_predictions
from vidlu.utils.tree import print_tree
from vidlu import defaults
from vidlu.factories import get_data

import dirs

# python view_dataset.py
#   mnist all
#   inaturalist2018 train
#   voc2012 test
#   wilddash bench

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--permute', action='store_true')
args = parser.parse_args()

ds = get_data(args.ds + '{' + args.part + '}', datasets_dir=dirs.DATASETS,
              cache_dir=dirs.CACHE)[0]
print("Name:", ds.name)
print("Info:")
print_tree(ds.info, depth=1)
print("Number of examples:", len(ds))
print(f"Size estimate: {pickle_sizeof(ds[0]) * len(ds) / 2 ** 30:.3f} GiB")

if 'class_count' not in ds.info:
    ds.info['class_count'] = 2

if args.augment:
    ds = ds.map(defaults.get_jitter(ds))

if args.permute:
    ds = ds.permute()

view_predictions(ds.map(lambda r: (image.to_pil(mode='RGB').to_numpy().item, r.y)),
                 infer=lambda x: ds[0][1])
