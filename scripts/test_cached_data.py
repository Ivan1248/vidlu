import argparse

import numpy as np
from tqdm import trange

from _context import vidlu

from dl_uncertainty import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('--trainval', action='store_true')
args = parser.parse_args()
print(args)

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_train, ds_test = vidlu.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

# Model

for i in trange(len(ds_test)):
    for j in range(i + 1, len(ds_test)):
        assert (ds_test[i][0] != ds_test[j][0]).any(), (i, j)

for i in trange(len(ds_train)):
    for j in range(i + 1, len(ds_train)):
        assert (ds_train[i][0] != ds_train[j][0]).any(), (i, j)
