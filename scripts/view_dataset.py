import argparse

# noinspection PyUnresolvedReferences
import _context
from vidlu.utils.misc import pickle_sizeof
from vidlu.transforms import image
from vidlu.utils.presentation.visualization import view_predictions
from vidlu.utils.tree import print_tree
from vidlu.factories import prepare_dataset, get_data
from vidlu.utils import debug

import dirs

# python view_dataset.py
#   Cityscapes
#   Cityscapes --subset all
#   Cityscapes --subset train
#   "VOC2012Segmentation(pad=True,size_unit=32)" --subset val

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--jitter', type=str, default=None)
parser.add_argument('--permute', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

debug.set_traceback_format(call_pdb=args.debug, verbose=False)

subset_expr = f'{{{args.subset}}}' if args.subset else ''

[[ds], [name], _] = get_data(f"{args.ds}{subset_expr}", datasets_dir=dirs.datasets,
                             cache_dir=dirs.cache)

ds = prepare_dataset(ds)
if isinstance(name, tuple):
    name = f'({" ".join(name)})'
print("Name:", ds.identifier, name)
print("Info:")
print_tree(ds.info.dict_, depth=1)
print("Number of examples:", len(ds))
print(f"Size estimate: {pickle_sizeof(ds[0]) * len(ds) / 2 ** 30:.3f} GiB")

if 'class_count' not in ds.info:
    raise RuntimeError("ds.info.class_count not defined.")

if args.jitter:
    jitter = eval("jitter." + args.jitter)
    ds = ds.map(jitter)

if args.permute:
    ds = ds.permute()


def transform(r):
    x = image.torch_to_numpy(r[0].permute(1, 2, 0))
    if len(r) == 1:
        return x, 0
    return x, r[1].numpy()


ds = ds.map(transform)
view_predictions(ds, infer=None)
