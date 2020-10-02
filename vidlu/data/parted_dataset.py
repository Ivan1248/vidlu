import typing as T
from typing import Iterator

from vidlu.utils.func import valmap
from collections.abc import Mapping

from .dataset import Dataset


# PartitionedDataset


class _PartSplit:
    def __init__(self, subpart_names, ratio):
        self.subparts = subpart_names
        self.ratio = ratio


def _generate_parts(part_to_ds: T.Mapping[str, Dataset],
                    part_to_split: T.Mapping[str, T.Tuple[T.Tuple[str, str], float]] = None):
    part_to_split = {k: _PartSplit(*v) for k, v in part_to_split.items()}
    parts = set(part_to_ds.keys())
    parts.update(p for k, ps in part_to_split.items() for p in [k] + list(ps.subparts))
    part_to_ds = dict(part_to_ds)

    def generate_parts(part):
        if part not in part_to_split:
            return False
        subparts = part_to_split[part].subparts
        if part in part_to_ds:
            if any(s not in part_to_ds for s in subparts):
                if not all(s not in part_to_ds for s in subparts):
                    raise ValueError("If subparts are provided, either all or none of them"
                                     + " must be provided. E.g. trainval can be provided either"
                                     + " without any or with both of {train, val}.")
                (s, t), ratio = subparts, part_to_split[part].ratio
                part_to_ds[s], part_to_ds[t] = part_to_ds[part].split(ratio=ratio)
                return True
        elif all(s in part_to_ds for s in subparts):
            s, t = (part_to_ds[s] for s in subparts)
            part_to_ds[part] = s + t
            return True
        return False

    while any(map(generate_parts, parts)):  # TODO: optimize
        pass
    return part_to_ds


class PartedDataset(Mapping):
    def __init__(self, part_to_ds, part_to_split=None):
        part_to_split = part_to_split or {}
        non_top_level_parts = set(s for subsets, ratio in part_to_split.values() for s in subsets)
        self.top_level_parts = [k for k in part_to_split.keys() if k not in non_top_level_parts]
        self.part_to_ds = _generate_parts(part_to_ds, part_to_split)

    def __getitem__(self, item):
        try:
            return self.part_to_ds[item]
        except KeyError:
            raise KeyError(f'The parted dataset does not have a part called "{item}".')

    def __getattr__(self, item):
        return self[item]

    def __len__(self) -> int:
        return len(self.part_to_ds)

    def __iter__(self):
        yield from self.keys()

    def with_transform(self, transform):
        return PartedDataset(valmap(transform, self.part_to_ds))

    def keys(self):
        return self.part_to_ds.keys()

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def top_level_items(self):
        for k in self.top_level_parts:
            yield k, self[k]


"""
class CachingGetter:
    def __init__(self, get):
        assert parameter_count(get) == 0
        self._get, self._val = get, None

    def __call__(self):
        if self._val is None:
            self._val = self._get()
            del self._get
        return self._val

class PartedDatasetL:

    def __init__(self,
                 part_to_getter: Dict[str, Callable],
                 part_to_split: Dict[str, Tuple[Tuple[str, str], float]] = None):
        if part_to_split is None:
            part_to_split = {"all": (("trainval", "test"), 0.8),
                             "trainval": (("train", "val"), 0.8)}
        part_to_split = {k: PartSplit(*v) for k, v in
                         part_to_split.items()}
        parts = set()
        parts.update(part_to_getter.keys())
        for k, ps in part_to_split.items():
            parts.update(
                p for p in [k] + list(ps.subparts) for k, ps in part_to_split.items() if
                p not in parts)
        part_to_getter = {k: CachingGetter(v) for k, v in part_to_getter.items()}

        def generate_parts(part):
            if part not in part_to_split:
                return False
            if part in part_to_getter:
                subparts = part_to_split[part].subparts
                if any(s not in part_to_getter for s in subparts):
                    if not all(s not in part_to_getter for s in subparts):
                        raise ValueError("If subparts are provided, either all or none of them "
                                         "must be provided. E.g. trainval can be provided either "
                                         "without any or with both of {train, val}.")
                    (s, t), ratio = subparts, part_to_split[part].ratio
                    s_t_ = CachingGetter(lambda: part_to_getter[part]().split(ratio))
                    for i, k in enumerate([s, t]):
                        part_to_getter[k] = CachingGetter(lambda: (lambda i=i: s_t_()[i])())
                    return True
            elif all(s in part_to_getter for s in part_to_split[part].subparts):
                s, t = (part_to_getter[s] for s in part_to_split[part].subparts)
                part_to_getter[part] = CachingGetter(lambda: s() + t())
                return True
            return False

        while any(map(generate_parts, parts)):
            pass
        assert all(x in parts for x in chain(part_to_getter, part_to_split))
        self._part_to_getter = part_to_getter

    def __getitem__(self, item):
        return self._part_to_getter[item]()

    def __getattr__(self, item):
        return self[item]

    def part_names(self):
        return self._part_to_getter.keys()
"""
