from itertools import chain
from typing import Dict, Tuple
from functools import lru_cache

from .dataset import Dataset


# PartitionedDataset


class PartSplit:
    def __init__(self, subpart_names, ratio):
        self.subparts = subpart_names
        self.ratio = ratio


class DatasetSet:
    def __init__(self,
                 part_to_ds: Dict[str, Dataset],
                 part_to_split: Dict[str, Tuple[Tuple[str, str], float]] = None):
        if part_to_split is None:
            part_to_split = {"all": (("trainval", "test"), 0.8),
                             "trainval": (("train", "val"), 0.8)}
        part_to_split = {k: PartSplit(*v) for k, v in
                         part_to_split.items()}
        parts = set()
        parts.update(part_to_ds.keys())
        for k, ps in part_to_split.items():
            parts.update(
                p for p in [k] + list(ps.subparts) for k, ps in part_to_split.items() if
                p not in parts)
        part_to_ds = {k: v for k, v in part_to_ds.items()}

        def generate_parts(part):
            if part not in part_to_split:
                return False
            if part in part_to_ds:
                subparts = part_to_split[part].subparts
                if any(s not in part_to_ds for s in subparts):
                    if not all(s not in part_to_ds for s in subparts):
                        raise ValueError("If subparts are provided, either all or none of them " +
                                         "must be provided. E.g. trainval can be provided either " +
                                         "without any or with both of {train, val}.")
                    (s, t), ratio = subparts, part_to_split[part].ratio
                    part_to_ds[s], part_to_ds[t] = part_to_ds[part].split(ratio=ratio)
                    return True
            elif all(s in part_to_ds for s in part_to_split[part].subparts):
                s, t = (part_to_ds[s] for s in part_to_split[part].subparts)
                part_to_ds[part] = s + t
                return True
            return False

        while any(map(generate_parts, parts)):
            pass
        assert all(x in parts for x in chain(part_to_ds, part_to_split))
        self._part_to_ds = part_to_ds

    def __getitem__(self, item):
        try:
            return self._part_to_ds[item]
        except KeyError:
            raise KeyError(f'The dataset does not have a part called "{item}".')

    def __getattr__(self, item):
        return self[item]

    def with_transform(self, transform):
        return DatasetSetWithTransform(self, transform)

    def keys(self):
        return self._part_to_getter.keys()

    def items(self):
        for k in self.keys():
            yield k, self[k]


class DatasetSetWithTransform:
    def __init__(self, dataset_splits, transform):
        self.dss = dataset_splits
        self.get = lru_cache()(lambda item: transform(self.parted_dataset.__getitem__(item)))

    def __getitem__(self, item):
        return self.get(item)

    def __getattr__(self, item):
        return self.get(item)

    def keys(self):
        return self._dss.keys()


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
