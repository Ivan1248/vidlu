from collections.abc import Sequence

from vidlu.huml import slice_len


class LazySeq(Sequence):
    slots = ()

    def __init__(self, getitem, length):
        self._getitem, self._length = getitem, length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, _, step = idx.indices(len(self))
            return LazySeq(slice_len(idx, len(self)), lambda i: self._getitem(start + i * step))
        elif isinstance(idx, (Sequence,)):
            return LazySeq(len(idx), lambda i: self._getitem(idx[i]))
        return self._getitem(idx)

    def __len__(self):
        return self._length
