import inspect
import pickle
import typing as T

import numpy as np
from torch.utils.data.dataloader import default_collate as torch_collate

from .record import Record
import vidlu.data.types as dt


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
    elif isinstance(elem, T.Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, T.Sequence):
        return list(map(numpy_collate, zip(*batch)))
    else:
        raise TypeError(type(elem))


class ExtendableCollate:
    def __call__(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, T.Mapping):
            return {key: self([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self(samples) for samples in zip(*batch, strict=True)))
        elif isinstance(elem, T.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch, strict=True)
            return [self(samples) for samples in transposed]
        else:
            return torch_collate(batch)


class DefaultCollate(ExtendableCollate):
    def __call__(self, batch):
        elem_type = type(batch[0])
        if hasattr(elem_type, 'collate'):
            try:
                if len(inspect.signature(elem_type.collate).parameters) > 1:
                    return elem_type.collate(batch, self)
                else:
                    return elem_type.collate(batch)
            except NotImplementedError:
                pass
        if elem_type is Record:
            return Record(self(tuple(dict(d.items()) for d in batch)))
        elif isinstance(batch[0], dt.AABBsOnImage):
            return batch
        return super().__call__(batch)


default_collate = DefaultCollate()
