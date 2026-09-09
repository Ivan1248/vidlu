"""Tests for `default_prepare_batch`'s device placement.

The batch is a tree, not a flat mapping: a batch can carry one entry per input modality,
each entry a collection of tensors. Moving only the top level would leave those tensors on
the CPU and the step would fail on a device mismatch -- but only on a GPU, so the contract
is pinned here with the `meta` device, which makes any move visible on any machine.
"""

import typing as T

import pytest
import torch

from vidlu.data import BatchTuple
from vidlu.training.trainers import default_prepare_batch

DEVICE = torch.device("meta")


def prepared(batch):
    return default_prepare_batch(batch, device=DEVICE)


def devices(x):
    """The device of every tensor in `x`, in traversal order."""
    if isinstance(x, torch.Tensor):
        return [x.device]
    elif hasattr(type(x), "items"):
        return [d for v in x.values() for d in devices(v)]
    elif isinstance(x, (str, bytes)):
        return []
    elif isinstance(x, T.Sequence):
        return [d for v in x for d in devices(v)]
    return []


def a_tensor():
    return torch.zeros(2, 3)


# ----- moving ---------------------------------------------------------------------

def test_a_bare_tensor_batch_is_moved():
    assert prepared(a_tensor()).device == DEVICE


def test_top_level_entries_are_moved():
    out = prepared(dict(x=a_tensor(), y=a_tensor()))
    assert devices(out) == [DEVICE] * 2


@pytest.mark.parametrize("nested", [
    pytest.param(lambda: {"feats": a_tensor(), "logits": a_tensor()}, id="mapping"),
    pytest.param(lambda: [a_tensor(), a_tensor()], id="list"),
    pytest.param(lambda: (a_tensor(), a_tensor()), id="tuple"),
    pytest.param(lambda: BatchTuple([a_tensor(), a_tensor()]), id="BatchTuple"),
])
def test_a_nested_collection_is_moved_too(nested):
    """One entry per input modality is a collection, not a tensor; moving only the top
    level would leave its tensors behind."""
    out = prepared(dict(data=nested(), target=a_tensor()))
    assert devices(out) == [DEVICE] * 3


def test_a_nested_collection_keeps_its_type():
    out = prepared(dict(data=BatchTuple([a_tensor()]), ids=["a"], target=a_tensor()))
    assert type(out["data"]) is BatchTuple


# ----- passing through ------------------------------------------------------------

@pytest.mark.parametrize("value", [7, object()], ids=["int", "object"])
def test_a_leaf_that_cannot_be_moved_passes_through(value):
    """Batches legitimately carry metadata alongside the tensors."""
    assert prepared(dict(target=a_tensor(), meta=value))["meta"] is value


def test_strings_inside_a_collection_survive():
    """`str` is a `Sequence`, so recursing into one would rebuild it from a generator and
    replace each id with the generator's repr."""
    ids = ["road-1", "road-2"]
    assert prepared(dict(target=a_tensor(), segment_ids=ids))["segment_ids"] == ids


# ----- validation -----------------------------------------------------------------

def test_an_unsupported_batch_is_rejected():
    """Only the batch itself is type-checked: this catches a caller passing the wrong
    thing, while entries stay free to hold metadata."""
    with pytest.raises(TypeError, match="Invalid batch type"):
        prepared(7)


def test_a_string_batch_is_rejected():
    """A `str` is a `Sequence`, so it would otherwise be mistaken for a batch of leaves."""
    with pytest.raises(TypeError, match="Invalid batch type"):
        prepared("not a batch")


# ----- the traversal shared with `untag` ------------------------------------------

def test_untag_reaches_tensors_nested_in_the_batch():
    """`untag` walks the same tree as `default_prepare_batch`; a modality mapping would
    otherwise reach the model still wearing its `DataModality` subclass."""
    from vidlu.data.types import Array
    from vidlu.training.steps import untag

    tagged = a_tensor().as_subclass(Array)
    out = untag(dict(data={"feats": tagged}, ids=["a"], target=tagged))

    assert type(out["data"]["feats"]) is torch.Tensor
    assert type(out["target"]) is torch.Tensor
    assert out["ids"] == ["a"]


def test_map_tensors_leaves_the_original_alone():
    """Both callers document that the input is unchanged."""
    from vidlu.torch_utils import map_tensors

    inner = a_tensor()
    batch = dict(data={"feats": inner})
    out = map_tensors(batch, lambda t: t + 1)

    assert out["data"]["feats"] is not inner
    assert torch.equal(inner, torch.zeros(2, 3))
