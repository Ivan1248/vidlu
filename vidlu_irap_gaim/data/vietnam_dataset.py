"""IRAP-Vietnam dataset factory.

Thin wrapper around :func:`vidlu_irap_gaim.data.bih_dataset.make_bih_data`
that defaults to the Vietnam paths and disables the BiH N-context pickle-based
filter (Vietnam metadata does not ship those pickles by default — see
``.devdocs/irap_vietnam_data_preparation.md``, simplification 1).

Default-path resolution
-----------------------
When ``dataset_dir`` is not given, ``IRAP_Vietnam`` is searched (in order) under:

1. ``$IRAP_HOME/IRAP_Vietnam`` (if ``IRAP_HOME`` is set).
2. ``$DATASETS_PATH/IRAP_Vietnam`` (if ``DATASETS_PATH`` is set).
3. The first ancestor of this file that contains ``data/datasets/IRAP_Vietnam``.

When ``metadata_dir`` is not given, it defaults to ``<dataset_dir>`` (metadata
files live directly in the dataset root).

The underlying dataset class :class:`BihSequence` is reused unchanged: the
Vietnam metadata directory mirrors the BiH layout, and segment ids are
consecutive integers within each section so the integer-arithmetic context
resolution works as in BiH.
"""

from __future__ import annotations

import os
from pathlib import Path
import typing as T

from vidlu.utils.path import find_in_ancestors

from .bih_dataset import make_bih_data


DATASET_NAME = "IRAP_Vietnam"


def _resolve_default_dataset_dir() -> Path:
    """Find ``IRAP_Vietnam`` in standard locations. See module docstring."""
    candidates: list[Path] = []
    if v := os.environ.get("IRAP_HOME"):
        candidates.append(Path(v) / DATASET_NAME)
    if v := os.environ.get("DATASETS_PATH"):
        candidates.append(Path(v) / DATASET_NAME)
    try:
        candidates.append(Path(find_in_ancestors(__file__, f"data/datasets/{DATASET_NAME}")))
    except FileNotFoundError:
        pass
    for c in candidates:
        if c.is_dir():
            return c
    raise RuntimeError(
        f"Cannot find {DATASET_NAME!r}. Pass dataset_dir explicitly, or set "
        f"IRAP_HOME / DATASETS_PATH. Checked: "
        + (", ".join(str(c) for c in candidates) if candidates else "(no candidates)")
    )


def make_vietnam_data(
    *,
    dataset_dir: str | Path | None = None,
    context_sequence: T.Sequence[int] = (0, -1, -4),
    use_ncontext_filter: bool = False,
    allow_missing_attributes: bool = True,
    **kwargs,
):
    """Build IRAP-Vietnam {train, val, test} datasets.

    Differs from :func:`make_bih_data` only in defaults:

    - ``use_ncontext_filter=False``: the Vietnam release does not include the
      precomputed N-context pickles.
    - ``allow_missing_attributes=True``: five flow attributes (motorcycle /
      bicycle / pedestrian) are empty in every coding table — without this
      flag every segment would be dropped. Missing/unmappable codes are
      mapped to PyTorch's standard ``ignore_index = -1``.

    Default-path resolution for ``dataset_dir`` is documented in the module
    docstring; ``metadata_dir`` defaults to ``<dataset_dir>``. All other
    keyword arguments are forwarded to :func:`make_bih_data`.
    """
    assert 'metadata_dir' not in kwargs, "metadata_dir is fixed to dataset_dir in make_vietnam_data"

    dataset_dir = Path(dataset_dir) if dataset_dir is not None else _resolve_default_dataset_dir()
    metadata_dir = dataset_dir

    return make_bih_data(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        context_sequence=context_sequence,
        use_ncontext_filter=use_ncontext_filter,
        allow_missing_attributes=allow_missing_attributes,
        **kwargs,
    )
