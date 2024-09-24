from collections import Counter

import pytest
from collections.abc import Sequence

import numpy as np
import torch

from vidlu.data import Record, Dataset, DataLoader


class TestRecord:
    def test_record_comparison(self):
        r = Record(a_=lambda: 2 + 3, b=7)
        assert r.b == 7
        assert str(r) == "Record(a=<unevaluated>, b=7)"
        assert not r.is_evaluated('a')
        assert r.a == 5
        assert r.is_evaluated('a')
        assert r == Record(r)
        assert r == Record({'a': 5, 'b': 6}, b=7)
        assert r == Record(zip(['a', 'b'], [5, 7]))
        assert r == Record(a_=lambda: 5, b_=lambda: 7)
        assert r == Record(list(r.items()))
        assert list(r.items()) == [('a', 5), ('b', 7)]
        assert Record(a=5, b_=lambda: 7) != Record(a=5, b_=lambda: 42)
        assert Record(a=5, b_=lambda: 7) == Record(a_=lambda: 5, b=7)

    def test_record_indexing(self):
        r = Record(a_=lambda: 2 + 3, b=7)
        a, b = r
        assert all(a == b for a, b in zip(r.items(), zip(r.keys(), r.values())))
        assert a == r.a and b == r.b
        for _ in r.values():
            pass
        assert r.a == r['a'] == r[0]

    def test_record_str(self):
        r2 = Record(b_=lambda: 4, c_=lambda: 8)
        assert str(r2) == "Record(b=<unevaluated>, c=<unevaluated>)"
        r3 = Record(r2, c=1, d=2)
        assert str(r3) == "Record(b=<unevaluated>, c=1, d=2)"
        assert str(r3[('d', 'b')]) == "Record(d=2, b=<unevaluated>)"

    def test_record_join(self):
        with pytest.raises(Exception):  # missing overwrite=True
            Record(a=1, b=2).join(Record(a_=lambda: 3, c=4))
        r4 = Record(a=1, b=2).join(Record(a_=lambda: 3, c=4), overwrite=True)
        assert str(r4) == "Record(a=<unevaluated>, b=2, c=4)"
        r4.evaluate()
        assert str(r4) == "Record(a=3, b=2, c=4)"

    def test_record_reflexive(self):
        r5 = Record(a=2, b_=lambda: 3, c_=lambda r: r.a * r.b, d_=lambda r: r.c - r.b)
        r5.evaluate()
        assert str(r5) == "Record(a=2, b=3, c=6, d=3)"


class TestDataset:
    def test_dataset_length(self):
        ds = Dataset(name="SomeDataset", data=list(range(5)))
        assert type(ds[0]) is int
        assert ds[-1] == ds[len(ds) - 1]
        assert len(ds[-4:-1]) == 3
        assert ds[-4:][0] == ds[-4] == len(ds) - 4
        assert len(ds[-1::2]) == 1
        assert len(ds[-1:0:2]) == 0
        assert len(ds[:]) == len([x for x in ds[:-2]]) + 2 == len(ds)
        assert len(ds[:-1]) == len(ds) - 1
        assert len(ds[0::2]) == len(ds) // 2 + len(ds) % 2
        assert len(ds[1::2]) == len(ds) // 2

    def test_dataset_record_map(self):
        for t in [dict, Record]:
            data = [t(x=x, y=x ** 2) for x in range(20)]
            ds = Dataset(name=f"ds_{t.__name__}", data=data)
            assert len(ds) == len(data)
            for a in ds:
                assert a['y'] == a['x'] ** 2
            assert repr(ds) == str(ds)
            assert len(ds[0]) == 2
            assert ds.identifier == ds.name

        mds = ds.map(lambda a: Record(x=a.x, y=a.y ** 0.5), info={**ds.info, 'len': 20})
        assert mds.data == ds and 'len' in mds.info
        assert len(mds.identifier) > len(ds.identifier)
        assert mds.identifier[:len(ds.identifier)] == ds.identifier
        assert mds.name == mds.identifier[-len(mds.name):]
        for x, y in mds:
            assert y == (x ** 2) ** 0.5

    def test_dataset_funky_indexing(self):
        for t in [dict, Record]:
            data = [t(x=x, y=x ** 2, z=1) for x in range(20)]
            ds = Dataset(name=f"ds_{t.__name__}", data=data)

            mds = ds[:3, ['y', 'z']]
            if t is Record:
                assert list(mds) == list(ds[:3, 1:])

    def test_dataset_split_join(self):
        ds = Dataset(name="SomeDataset2", data=list(range(5)))
        assert len(ds.filter(lambda x: x < 2)) == 2

        ds1, ds2 = ds.repeat(4).split(ratio=0.5)
        assert len(ds1) == len(ds2) and all(a == b for a, b in zip(ds1, ds2))
        ds1, ds2 = ds.repeat(4).split(index=4 * len(ds) // 2)
        assert len(ds1) == len(ds2) and all(a == b for a, b in zip(ds1, ds2))
        ds1, ds2 = ds[:3].split(ratio=0.5)
        assert len(ds1) + len(ds2) == len(ds[:3]) == 3
        assert len(ds + ds) == 2 * len(ds)
        assert all(a == b for a, b in zip(ds1 + ds2, ds1.join(ds2)))
        assert all(a == b for a, b in zip(ds1 + ds2, ds[:3]))

    def test_dataset_zip_record_join(self):
        dsa = Dataset(name="ds", data=[Record(a=i, b=2 * i) for i in range(8)])
        dsb = Dataset(name="ds", data=[Record(c=4 * i) for i in range(8)])
        dsab = dsa.zip(dsb).map(lambda x: x[0].join(*x[1:]))
        assert len(dsab[0]) == len(dsa[0]) + len(dsb[0])
        assert all(x['c'] == 2 * x['b'] == 4 * x['a'] for x in dsab)

    def test_dataset_iter_slice(self):
        ds = Dataset(name="Pairs", data=[Record(x=i, y='y') for i in range(10)])

        assert len(list(ds)) == len(ds)
        assert [i for i, r in enumerate(ds)][-1] == len(ds) - 1

        assert len(ds[1::3]) == len([x for x in ds[1::3]])

    def test_info_cache_hdd(self, tmpdir):
        upper = 9
        for i in range(2):
            ds = Dataset(name="Pairs", data=[i for i in range(upper + 1)])
            call_counts = Counter()

            def call_counting(func):
                def apply(ds):
                    nonlocal call_counts
                    call_counts[func] += 1
                    return func(ds)

                return apply

            name_to_func = {'sum': call_counting(sum), 'max': call_counting(max)}

            ds = ds.info_cache_hdd(name_to_func, tmpdir)

            if i == 0:  # in the second run, the info is cached in the filesystem
                assert len(call_counts) == 0  # unless the debugger has already called ds.info
                _ = ds.info['sum']
                assert all(x == 1 for x in call_counts.values()) and len(call_counts) > 0
                _ = ds.info['sum']
                assert all(x == 1 for x in call_counts.values())
            else:
                _ = ds.info
                assert len(call_counts) == 0
                _ = ds.info
                assert len(call_counts) == 0

    def test_info_cache_hdd_parallel(self, tmpdir):
        lower = 5
        ds = Dataset(name="Numbers", data=[i + lower for i in range(100)])
        name_to_func = {'first': first, 'also': first}
        ds = ds.info_cache_hdd(name_to_func, tmpdir)
        ds = ds.map(SubFirst(ds))
        dl = DataLoader(ds, num_workers=3)
        assert min(x for x in dl) == 0
        assert ds.info.first == lower


# test_cache_lazy_info_hdd_parallel

class SubFirst:
    def __init__(self, ds):
        self.ds = ds

    def __call__(self, d):
        return d - self.ds.info.first


def first(ds):
    return ds[0]
