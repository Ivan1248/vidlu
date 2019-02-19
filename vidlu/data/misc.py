import pickle
import io
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate as torch_collate

from functools import partialmethod
from .record import Record


# Serialization ####################################################################################

def serialize(obj):
    with io.BytesIO() as b:
        pickle.dump(obj, b)
        return b.getvalue()


def serialized_sizeof(obj):
    with io.BytesIO() as b:
        pickle.dump(obj, b)
        return len(b.getbuffer())


# Collate ##########################################################################################

def default_collate(batch, array_type='torch'):
    collate = torch_collate if array_type == 'torch' else numpy_collate
    if type(batch[0]) is Record:
        return Record(collate(tuple(dict(d.items()) for d in batch)))
    return collate(batch)


def numpy_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__module__ == 'numpy':
        if elem.shape == ():  # scalars
            return np.array(batch)
        else:
            return np.stack(batch, 0)
    elif isinstance(elem, (int, float)):
        return np.array(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        return list(map(numpy_collate, zip(*batch)))
    else:
        raise TypeError(type(elem))


# DataLoader class with collate function suporting Record examples

class DataLoader(torch.utils.data.DataLoader):
    __init__ = partialmethod(torch.utils.data.DataLoader.__init__, shuffle=True,
                             collate_fn=default_collate, drop_last=True)
