from functools import partial

import numpy as np
import pytest

from vidlu.data.utils import zip_data_loader
from vidlu.data import DataLoader


class TestDataLoaders:
    def test_zip_dataloader(self):
        batch_size = 3
        for ns in [[9, 5], [7, 11, 13]]:
            primary_index_to_n = {'shortest': min(ns), 'longest': max(ns), **dict(enumerate(ns))}
            dss = [list(range(n)) for n in ns]
            loader_f = partial(zip_data_loader, num_workers=0, batch_size=batch_size,
                               collate_fn=lambda b: b)
            for primary_index, n in primary_index_to_n.items():
                loader = loader_f(*dss, primary_index=primary_index)
                info = dict(ns=ns, primary_index=primary_index, items=list(loader))
                assert len(list(loader)) == len(loader), info
                assert len(loader) == n // batch_size, info
            with pytest.raises(RuntimeError):
                loader_f(*dss, primary_index='equal')
