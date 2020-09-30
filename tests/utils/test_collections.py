import pytest

import copy

import vidlu.utils.collections as vuc


class TestNameDict:
    def test_namedict(self):
        nd = vuc.NameDict(dict(a=5), b=6, c=7)
        assert len(nd) == 3
        assert nd.a == nd['a']
        with pytest.raises(AttributeError):
            nd.d
        with pytest.raises(KeyError):
            nd['d']
        nd.d = 8
        nd_copy = copy.deepcopy(nd)
        assert nd_copy == nd
