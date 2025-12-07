import inspect
import typing as T
import dataclasses as dc
import numpy as np
from torch.utils.data.dataloader import default_collate as torch_collate

from .record import Record, LazyItem
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


class MultiSizeBatch(list):
    pass


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
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                return MultiSizeBatch(batch)
            try:
                transposed = zip(*batch, strict=True)
            except TypeError as e:
                transposed = zip(*batch)
            return elem_type([self(samples) for samples in transposed])
        else:
            return torch_collate(batch)


@dc.dataclass
class DefaultCollate(ExtendableCollate):
    evaluate_lazy_data: bool = False

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
            batch_dicts = tuple((dict(r.items()) for r in batch) if self.evaluate_lazy_data else
                                (r.dict_ for r in batch))
            return Record(self(batch_dicts))
        elif elem_type is LazyItem:
            return LazyItem(lambda: self(tuple(x() for x in batch)))
        elif isinstance(batch[0], dt.AABBsOnImage):
            return batch
        return super().__call__(batch)


default_collate = DefaultCollate()
