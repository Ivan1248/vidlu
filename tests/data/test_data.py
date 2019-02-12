import pytest
from collections.abc import Sequence

import numpy as np
import torch

from vidlu.data import Record, Dataset, PartedDataset


class TestData:
    def test_record(self):
        r = Record(a_=lambda: 2 + 3, b=7)
        assert r.b == 7
        assert str(r) == "Record(a=<unevaluated>, b=7)"
        assert not r.is_evaluated('a')
        assert r.a == 5
        assert r.is_evaluated('a')
        assert r == Record(r)
        assert r == Record({'a': 5, 'b': 6}, b=7)
        assert r == Record(a_=lambda: 5, b_=lambda: 7)
        assert r == Record(list(r.items()))
        assert list(r.items()) == [('a', 5), ('b', 7)]
        a, b = r
        assert all(a == b for a, b in zip(r.items(), zip(r.keys(), r.values())))
        assert a == r.a and b == r.b
        for k in r.values():
            pass
        assert r.a == r['a'] == r[0]
        r2 = Record(b_=lambda: 4, c_=lambda: 8)
        assert str(r2) == "Record(b=<unevaluated>, c=<unevaluated>)"
        r3 = Record(r2, c=1, d=2)
        assert str(r3) == "Record(b=<unevaluated>, c=1, d=2)"
        assert str(r3[('d', 'b')]) == "Record(d=2, b=<unevaluated>)"
        with pytest.raises(Exception):
            Record(a=1, b=2).join(Record(a_=lambda: 3, c=4))
        r4 = Record(a=1, b=2).join(Record(a_=lambda: 3, c=4), overwrite=True)
        assert str(r4) == "Record(a=<unevaluated>, b=2, c=4)"
        r4.evaluate()
        assert str(r4) == "Record(a=3, b=2, c=4)"
        assert Record(a=5, b_=lambda: 7) != Record(a=5, b_=lambda: 42)
        assert Record(a=5, b_=lambda: 7) == Record(a_=lambda: 5, b=7)

    def test_dataset(self):
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

        for t in [dict, Record]:
            data = [t(x=x, y=x ** 2) for x in range(20)]
            ds = Dataset(name=f"ds_{t.__name__}", data=data)
            assert len(ds) == len(data)
            for a in ds:
                assert a['y'] == a['x'] ** 2
            assert repr(ds) == str(ds)
            assert len(ds[0]) == 2

        mds = ds.map(lambda a: Record(x=a.x, y=a.y ** 0.5), info={**ds.info, 'len': 20})
        assert mds.data == ds and 'len' in mds.info
        assert len(mds.full_name) > len(ds.full_name) and mds.name[
                                                          :len(ds.full_name)] == ds.full_name
        for x, y in mds:
            assert y == (x ** 2) ** 0.5

        ds = Dataset(name="SomeDataset2", data=list(range(5)))
        assert len(ds.filter(lambda x: x < 2)) == 2

        ds1, ds2 = ds.repeat(4).split(ratio=0.5)
        assert len(ds1) == len(ds2) and all(a == b for a, b in zip(ds1, ds2))
        ds1, ds2 = ds.repeat(4).split(position=4 * len(ds) // 2)
        assert len(ds1) == len(ds2) and all(a == b for a, b in zip(ds1, ds2))
        ds1, ds2 = ds[:3].split(ratio=0.5)
        assert len(ds1) + len(ds2) == len(ds[:3]) == 3
        assert len(ds + ds) == 2 * len(ds)
        assert all(a == b for a, b in zip(ds1 + ds2, ds1.join(ds2)))
        assert all(a == b for a, b in zip(ds1 + ds2, ds[:3]))

        dsa = Dataset(name="Numbers", data=list(range(10)))
        dsb = dsa.map(lambda x: -x, func_name="neg")
        dsab = dsa.zip(dsb, dsb)
        assert len(dsab) == len(dsa)
        assert type(dsab[0]) is tuple
        assert all(x[0] == -x[1] == -x[2] for x in dsab)
        for name, t in [("numpy", np.ndarray), ("torch", torch.Tensor)]:
            dsab_coll = dsab.collate(name)
            assert type(dsab_coll[0]) is t and dsab_coll[0].shape == (3,)
            dsab_coll2 = dsab.zip(dsab).collate(name)
            assert isinstance(dsab_coll2[0], Sequence) and type(dsab_coll2[0][0]) is t
            elem = (np.stack if "numpy" else torch.stack)(dsab_coll2[0])
            assert elem.shape == (3, 2)

        dsa = Dataset(name="ds", data=[Record(a=i, b=2 * i) for i in range(8)])
        dsb = Dataset(name="ds", data=[Record(c=4 * i) for i in range(8)])
        dsab = dsa.zip(dsb).map(lambda x: x[0].join(*x[1:]))
        assert len(dsab[0]) == len(dsa[0]) + len(dsb[0])
        assert all(x['c'] == 2 * x['b'] == 4 * x['a'] for x in dsab)

        ds = Dataset(name="Pairs", data=[Record(x=i, y=2 * i) for i in range(10)])
        ds_b = ds.batch(3)
        assert len(ds_b) == len(ds_b[:])
        assert len(ds_b) == (len(ds) + 3 - 1) // 3
        assert all(len(b) == 3 for b in ds_b[:-1]) and len(ds_b[-1]) == 1

    def test_parted_dataset(self):
        part_to_split = {
            'all': (('trainval', 'test'), 0.8),
            'trainval': (('train', 'val'), 0.7),
            'valtest': (('val', 'test'), 0.8 / 1.8),
        }
        for subsets in [["trainval", "test"], ["train", "val", "test"], ["all"]]:
            subset_to_getter = {name: Dataset(name=name, data=range(i * 10))
                                for i, name in enumerate(subsets)}
            pds = PartedDataset(subset_to_getter, part_to_split)

            def run_tests(pds):
                assert len(pds.trainval) == len(pds.train) + len(pds.val)
                assert all(a == b for a, b in zip(pds.trainval, pds.train + pds.val))
                assert len(pds.all) == len(pds.trainval) + len(pds.test)
                assert all(a == b for a, b in zip(pds.all, pds.trainval + pds.test))
                assert len(pds.valtest) == len(pds.all) - len(pds.train)

            run_tests(pds)
            pds_t = pds.with_transform(lambda x: x)
            run_tests(pds_t)
