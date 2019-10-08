import pickle
from collections.abc import Mapping, Sequence

import numpy as np
from torch.utils.data.dataloader import default_collate as torch_collate

from .record import Record


# Serialization ####################################################################################

def pickle_sizeof(obj):
    """An alternative to `sys.getsizeof` which works for lazily initialized objects (e.g. objects of
    type `vidlu.data.Record`) that can be much larger when pickled.

    Args:
        obj: the object to be pickled.

    Returns:
        int: the size of the pickled object in bytes.
    """
    return len(pickle.dumps(obj))


# Collate ##########################################################################################

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


def default_collate(batch, array_type='torch'):
    """A function Like `torch.utils.data.dataloader.default_collate`, but also
    supports the `vidlu.data.Record` type as a container."""
    collate = torch_collate if array_type == 'torch' else numpy_collate
    if type(batch[0]) is Record:
        return Record(collate(tuple(dict(d.items()) for d in batch)))
    return collate(batch)
