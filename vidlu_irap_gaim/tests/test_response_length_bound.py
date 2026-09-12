"""Tests for `ResponseScheme.max_response_tokens`.

The point of the bound is that it is *derived*, not tuned: the value set per
attribute is closed, so the longest legal response is computable and a compliant
model cannot exceed it.  These tests pin that property rather than any particular
number, and check the per-scheme worst-case choices (indices vs value strings,
sparse omitting defaults).

Pure unit tests driven by a whitespace tokenizer – no weights, no GPU.
"""

import torch

from vidlu_irap_gaim.vlm.response_scheme import make_response_scheme


class _WordTokenizer:
    """One token per whitespace-separated word, so counts are hand-checkable."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


# "Area type" has a deliberately long value so the worst case is identifiable.
ATTRS = {
    "Area type": {"Urban": 0, "Rural": 1, "Undivided rural road with wide verge": 2},
    "Lanes": {"One": 0, "Two": 1},
}
ATTR_NAMES = list(ATTRS)


def _scheme(name, **kwargs):
    return make_response_scheme(name, ATTRS, **kwargs)


def _bound(scheme, attrs=ATTR_NAMES):
    return scheme.compute_max_response_tokens(_WordTokenizer(), attrs)


# ----- the bound actually bounds ----------------------------------------------

def _all_legal_targets():
    """Every legal (class_idx per attribute) combination."""
    for a in range(len(ATTRS["Area type"])):
        for b in range(len(ATTRS["Lanes"])):
            yield torch.tensor([a, b])


def test_no_legal_response_exceeds_the_bound():
    """The property the whole design rests on, checked exhaustively over the
    closed output space rather than by sampling."""
    tokenizer = _WordTokenizer()
    for name in ("standard", "indexed", "json"):
        scheme = _scheme(name)
        bound = _bound(scheme)
        for target in _all_legal_targets():
            text = scheme.format_ground_truth(target, ATTR_NAMES)
            assert len(tokenizer.encode(text)) <= bound, f"{name} exceeded at {target}"


def test_sparse_bound_covers_the_nothing_omitted_case():
    """A sparse response is shortest when everything is default, so the bound has
    to be computed from the opposite end."""
    defaults = {"Area type": 0, "Lanes": 0}
    tokenizer = _WordTokenizer()
    for name in ("sparse_standard", "sparse_indexed"):
        scheme = _scheme(name, attr_to_default_class_idx=defaults)
        bound = _bound(scheme)
        for target in _all_legal_targets():
            text = scheme.format_ground_truth(target, ATTR_NAMES)
            assert len(tokenizer.encode(text)) <= bound, f"{name} exceeded at {target}"


# ----- worst-case selection is scheme-appropriate -----------------------------

def test_standard_bound_reflects_the_longest_value_string():
    """Dropping the long value shrinks the bound: the scheme picked it as worst
    case rather than using an arbitrary index."""
    long_value_bound = _bound(_scheme("standard"))
    short = {"Area type": {"Urban": 0, "Rural": 1}, "Lanes": {"One": 0, "Two": 1}}
    short_bound = make_response_scheme("standard", short).compute_max_response_tokens(
        _WordTokenizer(), list(short)
    )
    assert long_value_bound > short_bound


def test_indexed_bound_is_insensitive_to_value_string_length():
    """The indexed response carries digits, not value text, so a longer value
    name must not inflate the bound."""
    short = {"Area type": {"Urban": 0, "Rural": 1, "X": 2}, "Lanes": {"One": 0, "Two": 1}}
    assert _bound(_scheme("indexed")) == make_response_scheme(
        "indexed", short
    ).compute_max_response_tokens(_WordTokenizer(), list(short))


def test_indexed_bound_is_below_standard():
    """The reason indexed schemes are the cheap lever against truncation."""
    assert _bound(_scheme("indexed")) < _bound(_scheme("standard"))


# ----- subsetting -------------------------------------------------------------

def test_bound_scales_with_the_requested_attribute_subset():
    """Chunked requests must not be charged for attributes they do not ask for."""
    scheme = _scheme("standard")
    assert _bound(scheme, ["Lanes"]) < _bound(scheme, ATTR_NAMES)
