import torch
import typing as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import _context
from vidlu.factories import get_data
from vidlu.data import Dataset
from vidlu.ops import one_hot

import dirs

key_to_ds = get_data("CamVid{train, val, test}", dirs.DATASETS)
ds_train, ds_val, ds_test = (ds for k, ds in key_to_ds)


def class_incidence(y, c):
    y_oh = one_hot(y.unsqueeze(0).to(torch.int64).clamp_(0, c), c, torch.int64)
    return y_oh.squeeze(0).sum((0, 1))


c = ds_train.info.class_count
fig, ax = plt.subplots(3, sharey=True)
for i, (k, ds) in enumerate(key_to_ds):
    h = torch.zeros(c + 1)
    for x, y in tqdm(ds.map_fields(dict(y=torch.from_numpy))):
        h += class_incidence(y + 1, c + 1)
    h /= h.sum()
    h = h.numpy()
    ax[i].bar(np.arange(c + 1), h)
    for x, v in enumerate(h):
        ax[i].text(x - 0.25, v, f"{int(v + .5)}")
    ax[i].set_title(" ".join(k))
    ax[i].set_xticks([])
plt.xticks(np.arange(c + 1), ['Void'] + ds.info.class_names[:-1], size='small')
plt.show()
