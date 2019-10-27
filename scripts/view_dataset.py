import argparse

import _context
from vidlu.data.misc import pickle_sizeof
from vidlu.transforms import image, jitter
from vidlu.utils.presentation.visualization import view_predictions
from vidlu.utils.tree import print_tree
from vidlu.factories import get_prepared_data_for_trainer

import dirs

# python view_dataset.py
#   mnist all
#   inaturalist2018 train
#   voc2012 test
#   wilddash bench

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--jitter', type=str, default=None)
parser.add_argument('--permute', action='store_true')
args = parser.parse_args()

ds = get_prepared_data_for_trainer(args.ds + '{' + args.part + '}', datasets_dir=dirs.DATASETS,
                                   cache_dir=dirs.CACHE)[args.part]

print("Name:", ds.name)
print("Info:")
print_tree(ds.info, depth=1)
print("Number of examples:", len(ds))
print(f"Size estimate: {pickle_sizeof(ds[0]) * len(ds) / 2 ** 30:.3f} GiB")

if 'class_count' not in ds.info:
    ds.info['class_count'] = 2

if args.jitter:
    jitter = eval("jitter." + args.jitter)
    ds = ds.map(jitter)
    
if args.permute:
    ds = ds.permute()

ds = ds.map(lambda r: (image.torch_to_numpy(r[0].permute(1, 2, 0)), r[1].numpy()))
view_predictions(ds, infer=lambda x: ds[0][1])
