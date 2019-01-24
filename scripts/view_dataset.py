import argparse

from _context import vidlu
from vidlu.data import datasets
from vidlu import data_utils
from vidlu.utils.presentation.visualization import view_predictions
from vidlu.utils.image.data_augmentation import random_fliplr, augment_cifar

# python view_dataset.py
#   voc2012 test
#   cityscapes val
#   mozgalo train

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('part', type=str)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--permute', action='store_true')
args = parser.parse_args()

if args.ds == 'wilddash':
    names = ['val', 'bench']
    datasets = data_utils.get_cached_dataset_set_with_normalized_inputs(args.ds)
    datasets = dict(zip(names, datasets))
    ds = datasets[args.part]
elif args.part.startswith('test'):
    ds = data_utils.get_cached_dataset_set_with_normalized_inputs(
        args.ds, trainval_test=True)[1]
else:
    ds_train, ds_val = data_utils.get_cached_dataset_set_with_normalized_inputs(
        args.ds, trainval_test=False)
    ds = {
        'train': ds_train,
        'val': ds_val,
        'trainval': ds_train + ds_val,
    }[args.part]

if 'class_count' not in ds.info:
    ds.info['class_count'] = 2

if args.augment:
    ds = ds.map(data_utils.get_default_augmentation_func(ds))

if args.permute:
    ds = ds.permute()

view_predictions(ds, infer=lambda x: ds[0][1])
