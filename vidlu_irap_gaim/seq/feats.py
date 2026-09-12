import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vidlu.utils.path import create_file_atomic

# Names of the packed feature store inside a feature directory. The leading underscore keeps
# them out of the `<segment_id>.npy` namespace.
PACKED_FEATS_NAME = "_packed_feats.npy"
PACKED_INDEX_NAME = "_packed_feats_index.json"
# Where a logit directory records which columns belong to which attribute, so that the
# directory describes itself rather than relying on the reader to rebuild the layout.
ATTRIBUTE_SLICES_NAME = "attribute_slices.json"

# Directories whose packed store this process has verified against the per-segment files.
# Verifying means globbing tens of thousands of files, and the sequential-enhancement pipeline
# opens each store once per attribute, so it is done once per directory. `extract_features`
# drops a directory it writes to, which is the only way a store can go stale within a process.
_verified_pack_dirs: set[Path] = set()


def pack_features(feat_dir) -> tuple[Path, Path]:
    """Pack per-segment `.npy` feature files into a single array and index file.

    Saves packed array to `_packed_feats.npy` and mapping to `_packed_feats_index.json`.
    If the packed array already exists and matches current segment IDs in `feat_dir`,
    returns existing paths without re-packing.

    Args:
        feat_dir: Directory containing individual `<segment_id>.npy` files.

    Returns:
        Tuple of `(packed_array_path, index_json_path)`.

    Raises:
        RuntimeError: If `feat_dir` contains no feature files or contains mixed dtypes.
    """
    feat_dir = Path(feat_dir)
    array_path, index_path = feat_dir / PACKED_FEATS_NAME, feat_dir / PACKED_INDEX_NAME
    if feat_dir.resolve() in _verified_pack_dirs and array_path.exists() and index_path.exists():
        return array_path, index_path
    # Unsorted; sorting is only needed if (re)packing. The packed store is excluded by its
    # leading underscore.
    paths = [p for p in feat_dir.glob("*.npy") if not p.name.startswith("_")]
    segment_ids = {p.stem for p in paths}  # already in hand from the glob, so nearly free
    if array_path.exists() and index_path.exists():
        try:
            with index_path.open() as f:
                index = json.load(f)
        except json.JSONDecodeError:
            index = None  # truncated by an interrupted write; rebuilt below
        if index is not None and set(index) == segment_ids:
            _verified_pack_dirs.add(feat_dir.resolve())
            return array_path, index_path

    paths = sorted(paths)
    if len(paths) == 0:
        raise RuntimeError(f"No per-segment feature files (`<segment_id>.npy`) in {feat_dir}.")

    first = np.load(paths[0]).reshape(-1)
    array = np.empty((len(paths), first.size), dtype=first.dtype)
    array[0] = first
    for i, path in enumerate(tqdm(paths[1:], desc="Packing features", unit="seg"), start=1):
        feats = np.load(path).reshape(-1)
        # Assigning into `array` would silently cast; a directory holding several dtypes means
        # features were extracted by runs that disagreed, so the pack cannot represent them.
        if feats.dtype != array.dtype:
            raise RuntimeError(
                f"{path.name} has dtype {feats.dtype}, but {paths[0].name} has {array.dtype}."
                f" The feature directory {feat_dir} mixes dtypes, so it was written by runs with"
                f" different `feat_dtype`. Delete it and extract again with one dtype.")
        array[i] = feats
    # Atomic writes: an interrupted save would otherwise leave a truncated file that later
    # runs read as valid.
    create_file_atomic(array_path, lambda f: np.save(f, array))
    index = {path.stem: i for i, path in enumerate(paths)}
    create_file_atomic(index_path, lambda f: json.dump(index, f), mode="w")
    _verified_pack_dirs.add(feat_dir.resolve())
    return array_path, index_path


def _dataset_segment_ids(dataset) -> list[str] | None:
    """Per-index segment IDs of `dataset` without loading any examples, or None.

    Uses `dataset.info["segment_ids"]`, which `IRAPDataset` exposes in index
    order and which survives example-mapping wrappers. Returns None when the
    info entry is absent or its length does not match (e.g. after subsetting),
    in which case segment IDs must be read from the examples themselves.
    """
    info = getattr(dataset, "info", None)
    # Look up only this key: converting the whole info mapping would force
    # evaluation of unrelated lazy entries.
    segment_ids = None if info is None else info.get("segment_ids")
    if segment_ids is None or len(segment_ids) != len(dataset):
        return None
    return list(segment_ids)


def extract_features(trainer, dataset, out_dir, *, logit_dir=None, dtype: str = "float16",
                     skip_existing: bool = True, desc: str | None = None) -> dict:
    """Runs the model over `dataset` and saves its per-segment outputs as `<dir>/<segment_id>.npy`.

    Features go to `out_dir`. When `logit_dir` is given, the per-attribute logits of the
    same forward pass go there too, concatenated into one vector per segment, with
    `attribute_slices.json` recording which columns belong to which attribute. The logits
    are what sequential enhancement consumes as its `labels` and `logits` inputs, and they
    are already computed here -- discarding them would mean a second pass over the dataset.

    Args:
        trainer: Vidlu trainer instance containing model and data loader configuration.
        dataset: Dataset yielding samples with model inputs and 'segment_id'.
        out_dir: Output directory for saved feature `.npy` files.
        logit_dir: Optional output directory for saved per-attribute logits.
        dtype: Numerical datatype for saved arrays ('float16' or 'float32').
        skip_existing: If True, skips segments that already have every requested output.
        desc: Optional progress bar description.

    Returns:
        Dictionary with count of saved and skipped segments.
    """
    if dtype not in ("float16", "float32"):
        raise ValueError(f"dtype must be 'float16' or 'float32', got {dtype!r}.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if logit_dir is not None:
        logit_dir = Path(logit_dir)
        logit_dir.mkdir(parents=True, exist_ok=True)

    def is_complete(sid: str) -> bool:
        """Whether every requested output already exists for `sid`.

        Both are required, so a cache written before logits were exported is
        re-extracted rather than left half-populated.
        """
        return ((out_dir / f"{sid}.npy").exists()
                and (logit_dir is None or (logit_dir / f"{sid}.npy").exists()))

    num_skipped = 0
    check_per_example = skip_existing  # False once the dataset is prefiltered below
    if skip_existing and (segment_ids := _dataset_segment_ids(dataset)) is not None:
        missing_indices = [i for i, sid in enumerate(segment_ids) if not is_complete(sid)]
        num_skipped = len(dataset) - len(missing_indices)
        if len(missing_indices) == 0:
            return dict(saved=0, skipped=num_skipped)
        if num_skipped > 0:
            dataset = dataset[missing_indices]
        check_per_example = False

    # New per-segment files are about to be written, so an existing packed store is stale.
    for directory in (out_dir, logit_dir):
        if directory is not None:
            _verified_pack_dirs.discard(directory.resolve())

    trainer.model.eval()
    dl = trainer.get_data_loader(dataset, batch_size=trainer.eval_batch_size or 32,
                                 drop_last=False, shuffle=False)

    from vidlu.training.steps import _unify_sup_batch

    progress = tqdm(total=len(dataset), desc=desc, unit="seg", disable=desc is None,
                    smoothing=0.05)
    num_saved = 0
    attribute_slices_written = False
    with torch.no_grad(), progress:
        for batch in dl:
            prepared = trainer.prepare_batch(batch)
            x = _unify_sup_batch(prepared)[0]
            try:
                per_attribute_logits, feats = trainer.model(x, return_features=True)
            except TypeError as e:
                raise RuntimeError(
                    "The model must support return_features=True for feature export.") from e
            feats = feats.detach().cpu().numpy().astype(np.dtype(dtype))
            logits = None
            if logit_dir is not None:
                logits = torch.cat([a.detach().float() for a in per_attribute_logits], dim=1)
                logits = logits.cpu().numpy().astype(np.dtype(dtype))
                # Written from the shapes actually saved, so the index cannot describe a
                # layout other than the one in the files.
                if not attribute_slices_written:
                    _write_attribute_slices(logit_dir, per_attribute_logits)
                    attribute_slices_written = True
            segment_ids_in_batch = batch["segment_id"]
            for i, sid in enumerate(segment_ids_in_batch):
                if check_per_example and is_complete(sid):
                    num_skipped += 1
                    continue
                np.save(out_dir / f"{sid}.npy", feats[i])
                if logits is not None:
                    np.save(logit_dir / f"{sid}.npy", logits[i])
                num_saved += 1
            progress.update(len(segment_ids_in_batch))
    return dict(saved=num_saved, skipped=num_skipped)


def _write_attribute_slices(logit_dir: Path, per_attribute_logits) -> None:
    """Records the `[start, end)` column range of each attribute in the packed logits."""
    slices, start = [], 0
    for logits in per_attribute_logits:
        end = start + int(logits.shape[1])
        slices.append([start, end])
        start = end
    path = logit_dir / ATTRIBUTE_SLICES_NAME
    content = json.dumps(slices)
    if path.exists() and path.read_text() == content:
        return
    create_file_atomic(path, lambda f: f.write(content), mode="w")


def read_attribute_slices(logit_dir) -> list[tuple[int, int]]:
    """The `[start, end)` column range of each attribute in a logit directory."""
    path = Path(logit_dir) / ATTRIBUTE_SLICES_NAME
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing, so the columns of the packed logits cannot be assigned to"
            f" attributes. It is written by `extract_features(..., logit_dir=...)`; extract"
            f" again with a `logit_dir`.")
    with path.open() as f:
        return [(int(start), int(end)) for start, end in json.load(f)]


def export_feats(exp, split: str = "train", feat_dir: str = "FEATS/train",
                 logit_dir: str | None = None, dtype: str = "float16"):
    """Exports per-segment base-model outputs for a dataset split.

    Args:
        exp: Vidlu experiment instance (`e` in test run mode).
        split: Split name or prefix in `exp.data` to export.
        feat_dir: Output directory for `<segment_id>.npy` feature files.
        logit_dir: Optional output directory for the per-attribute logits.
        dtype: Data type for saved arrays ('float16' or 'float32').

    Returns:
        Dictionary containing output directory and export statistics.
    """
    dataset = next((ds for name, ds in exp.data.items() if name.startswith(split)), None)
    if dataset is None:
        raise RuntimeError(f"No dataset starting with '{split}' found in experiment data.")
    result = extract_features(exp.trainer, dataset, feat_dir, logit_dir=logit_dir, dtype=dtype,
                              desc=f"Extracting outputs ({split})")
    return dict(saved_to=str(feat_dir), **result)
