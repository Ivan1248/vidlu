from collections import defaultdict

import pytest

from vidlu.utils import tree
from vidlu.utils.func import ArgTree

atree = dict(a=5,
             b=dict(c=7,
                    d=[5, '11']),
             e=dict(),
             f=dict(g=dict()))

flat = [(('a',), 5),
        (('b', 'c'), 7),
        (('b', 'd'), [5, '11']),
        (('e',), dict()),
        (('f', 'g'), dict())]


def test_flatten():
    assert tree.flatten(atree) == flat


def test_unflatten():
    assert tree.equals(tree.unflatten(flat, type(atree)), atree)


def test_flatten_unflatten():
    assert tree.equals(tree.unflatten(tree.flatten(atree), type(atree)), atree)
    assert tree.flatten(tree.unflatten(flat, type(atree))) == flat


@pytest.mark.parametrize("tree_type", [defaultdict, ArgTree])
def test_convert(tree_type):
    atree_c = tree.convert(atree, tree_type)
    atree_cc = tree.convert(atree_c, type(atree))
    assert tree.equals(atree_cc, atree)


@pytest.mark.parametrize("tree_type", [defaultdict, ArgTree])
def test_unflatten_convert(tree_type):
    assert tree.equals(tree.unflatten(flat, ArgTree),
                       tree.convert(atree, ArgTree, convert_empty_trees=False))


def test_copy_equals():
    copy = tree.copy(atree)
    assert tree.equals(atree, copy)
