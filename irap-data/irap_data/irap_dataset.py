import json
import os
import pickle
from pathlib import Path
import typing as T
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from .dataset import Dataset
from .image_utils import load_image_cv2, center_crop, hwc_to_chw_float_tensor

RGB_MEAN: tuple[float, float, float] = (0.53354913, 0.52727484, 0.48752149)
RGB_STD: tuple[float, float, float] = (0.20401913, 0.20417478, 0.25402164)
INPUT_DIM: tuple[int, int, int] = (384, 288, 3)

IGNORE_LABEL_INDEX: int = -1


class MetaFiles:
    """File names (relative to a dataset's metadata directory)."""

    ATTRIBUTE_METADATA = "attribute_metadata.json"
    SPLITS = "splits.json"
    SEGMENT_ID_TO_DATA_PATHS = "segment_id_to_data_paths_rel.json"
    SEGMENT_ID_TO_ROAD_DATA = "segment_id_to_road_data.json"
    ROAD_ID_TO_SEGMENT_ID_SEQUENCE = "road_id_to_segment_id_sequence.json"

def resolve_datasets_root() -> Path:
    if v := os.environ.get("IRAP_HOME"):
        return Path(v)
    elif v := os.environ.get("DATASETS_PATH"):
        return Path(v)
    else:
        raise RuntimeError(
            "Cannot resolve datasets root directory. Please set the IRAP_HOME or DATASETS_PATH environment variable."
        )


def resolve_irap_paths(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve IRAP dataset and metadata directories.

    Returns:
        Tuple of (dataset_dir, metadata_dir) as Path objects.
    """
    irap_home = resolve_datasets_root()
    if dataset_dir is None:
        dataset_dir = irap_home / "IRAP_BIH"
    if metadata_dir is None:
        metadata_dir = irap_home / "IRAP_BIH_METADATA"
    return Path(dataset_dir), Path(metadata_dir)


def load_ncontext_segment_ids(
    seg_to_res_path: T.Union[str, Path],
    road_sequences: T.Union[str, Path, dict[str, list[str]]],
    max_N: int = 10,
    splits: T.Sequence[str] = ("train", "val", "test"),
) -> set[str]:
    """Load segment IDs from precomputed result pickle files with N-context filtering.

    This replicates the filtering in the original train_local_rec.py which:
    1. Loads segment IDs from seg_to_res_path/{split}.pickle files PER SPLIT
    2. Applies N-context filtering PER SPLIT (segments must have max_N neighbors
       on each side, all within the SAME split)
    3. Returns the union of filtered segments from all splits

    IMPORTANT: The original applies N-context filtering separately for each split,
    meaning a train segment must have all 10+10 context neighbors also in the train
    split. This differs from filtering on combined splits.

    Args:
        seg_to_res_path: Directory containing {train,val,test}.pickle files.
        road_sequences: Either a path to road_id_to_segment_id_sequence.json,
            or a dict mapping road_id -> list of segment IDs.
        max_N: Context window size (segments must have max_N neighbors on each side).
        splits: Which splits to load segment IDs from.

    Returns:
        Set of segment IDs that pass the N-context filter.
    """
    seg_to_res_path = Path(seg_to_res_path)

    # Load road sequences
    if isinstance(road_sequences, dict):
        road_id_to_segment_id_sequence = road_sequences
    else:
        road_seq_path = Path(road_sequences)
        if not road_seq_path.exists():
            raise FileNotFoundError(
                f"Road sequence file not found: {road_seq_path}\nThis file is required for N-context filtering."
            )
        with open(road_seq_path, "r") as f:
            road_id_to_segment_id_sequence = json.load(f)

    def ncontext_filter_for_split(split_segment_ids: set[str]) -> set[str]:
        """Apply N-context filtering for a single split (matches NContextDataset.build_contexts)."""
        filtered = set()
        for road_id, segment_sequence in road_id_to_segment_id_sequence.items():
            n_segments = len(segment_sequence)
            for i in range(max_N, n_segments - max_N):
                current = segment_sequence[i]
                if current not in split_segment_ids:
                    continue
                before = segment_sequence[i - max_N : i]
                after = segment_sequence[i + 1 : i + 1 + max_N]
                # All context segments must also be in THIS split's segment set
                if all(seg_id in split_segment_ids for seg_id in before + after):
                    filtered.add(current)
        return filtered

    # Process each split separately (matching original train_local_rec.py behavior)
    total_from_pickles = 0
    filtered_segment_ids = set()

    for split in splits:
        pickle_path = seg_to_res_path / f"{split}.pickle"
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Precomputed results file not found: {pickle_path}\n"
                f"This file is required for ncontext_segment_id_subset filtering."
            )

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        if "segment_id_to_idx" not in data:
            raise ValueError(
                f"Pickle file {pickle_path} does not contain 'segment_id_to_idx' key. "
                f"Available keys: {list(data.keys())}"
            )

        split_segment_ids = set(data["segment_id_to_idx"].keys())
        total_from_pickles += len(split_segment_ids)

        # Apply N-context filtering for THIS split only
        split_filtered = ncontext_filter_for_split(split_segment_ids)
        filtered_segment_ids.update(split_filtered)

        print(
            f"[load_ncontext_segment_ids] {split}: {len(split_segment_ids)} from pickle, "
            f"{len(split_filtered)} after N-context filter"
        )

    print(
        f"[load_ncontext_segment_ids] Total: {total_from_pickles} from pickles, "
        f"{len(filtered_segment_ids)} after N-context filter (max_N={max_N})"
    )

    return filtered_segment_ids


def make_irap_data(
    *,
    dataset_dir: str | Path,
    metadata_dir: str | Path,
    context_offsets: T.Sequence[int] = (0, -1, -4),
    mean: T.Sequence[float] = RGB_MEAN,
    std: T.Sequence[float] = RGB_STD,
    input_dim_rgb: T.Sequence[int] = INPUT_DIM,
    transforms: T.Mapping[str, T.Callable] | None = None,
    ncontext_segment_id_subset: set[str] | None = None,
    use_ncontext_filter: bool = False,
    seg_to_res_path: str | Path | None = None,
    allow_missing_attributes: bool = False,
):
    """Build IRAP datasets from an explicit dataset + metadata directory.

    Generic loader shared by the per-release presets :func:`make_bih_data` and
    :func:`make_vietnam_data`. Splits are discovered from ``splits.json``, so any
    IRAP release loads by pointing ``dataset_dir`` / ``metadata_dir`` at it (with
    matching flags); nothing here is BiH-specific.

    Args:
        dataset_dir: Directory holding the image/data files.
        metadata_dir: Directory holding the metadata JSONs (``splits.json``, ...).
        context_offsets: Offsets for context frames, e.g., (0, -1, -4).
        mean: RGB channel means for normalization.
        std: RGB channel stds for normalization.
        input_dim_rgb: Target image dimensions (W, H, C).
        transforms: Custom transforms dict.
        ncontext_segment_id_subset: Explicit set of segment IDs to include.
        use_ncontext_filter: If True, apply N-context filtering using pickle
            files at seg_to_res_path (or metadata_dir/seg_to_res if seg_to_res_path is None).
            Set to False to skip N-context filtering entirely.
        seg_to_res_path: Path to directory containing precomputed result pickle files
            ({train,val,test}.pickle). If None and use_ncontext_filter=True, uses
            metadata_dir/seg_to_res. Only used if use_ncontext_filter=True.
        allow_missing_attributes: If True, map missing/unmappable attribute codes to
            ``IGNORE_LABEL_INDEX`` instead of dropping the segment.
    """
    dataset_dir, metadata_dir = Path(dataset_dir), Path(metadata_dir)

    # Discover the split names actually present in the metadata. Unlabeled
    # subsets (`unlabeled_train`, `unlabeled_val`, ...) appear only when the
    # prep pipeline wrote them; iterating the file rather than a hardcoded
    # tuple makes those splits available transparently.
    with open(metadata_dir / MetaFiles.SPLITS, "r") as f:
        split_names = list(json.load(f).keys())

    # Load segment ID filter from precomputed results if requested.
    # The pickle files only cover labeled segments, so this filter is applied
    # to labeled subsets only; unlabeled subsets skip it.
    if use_ncontext_filter and ncontext_segment_id_subset is None:
        if seg_to_res_path is None:
            seg_to_res_path = metadata_dir / "seg_to_res"
        road_seq_path = metadata_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE
        ncontext_segment_id_subset = load_ncontext_segment_ids(seg_to_res_path, road_seq_path)

    if transforms is None:
        target_wh = (int(input_dim_rgb[0]), int(input_dim_rgb[1]))
        per_kind: dict[str, T.Callable] = {}
        # Photometric jittering is handled in TrainerConfig; keep loading deterministic here.
        per_kind["rgb"] = lambda img, _twh=target_wh: hwc_to_chw_float_tensor(center_crop(img, _twh))
        transforms = {split: per_kind for split in split_names}

    out = {}
    for split in split_names:
        is_unlabeled = split.startswith("unlabeled")
        out[split] = IRAPDataset(
            dataset_dir,
            split,
            metadata_dir=metadata_dir,
            transforms=transforms.get(split) if isinstance(transforms, dict) else transforms,
            context_offsets=tuple(context_offsets),
            mean=tuple(float(x) for x in mean),
            std=tuple(float(x) for x in std),
            # Pickle-driven N-context filtering doesn't cover unlabeled segments.
            ncontext_segment_id_subset=None if is_unlabeled else ncontext_segment_id_subset,
            # Unlabeled segments have no row in segment_id_to_road_data.json;
            # without this they'd all be dropped.
            allow_missing_attributes=True if is_unlabeled else allow_missing_attributes,
        )
    return out


def make_bih_data(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    use_ncontext_filter: bool = True,
    **kwargs,
):
    """Build IRAP-BiH datasets (preset over :func:`make_irap_data`).

    Defaults ``dataset_dir`` / ``metadata_dir`` to ``<root>/IRAP_BIH`` /
    ``<root>/IRAP_BIH_METADATA`` and ``use_ncontext_filter=True`` (BiH ships the
    precomputed N-context pickles). All other keyword arguments are forwarded to
    :func:`make_irap_data`.
    """
    dataset_dir, metadata_dir = resolve_irap_paths(
        dataset_dir=dataset_dir, metadata_dir=metadata_dir
    )
    return make_irap_data(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        use_ncontext_filter=use_ncontext_filter,
        **kwargs,
    )


def make_vietnam_data(
    *,
    dataset_dir: str | Path | None = None,
    use_ncontext_filter: bool = False,
    allow_missing_attributes: bool = True,
    **kwargs,
):
    """Build IRAP-Vietnam datasets (preset over :func:`make_irap_data`).

    Returns a dict with one entry per key in ``splits.json``. Labeled splits
    are always ``train`` / ``val`` / ``test``. When the prep pipeline wrote
    unlabeled images, the dict additionally contains ``unlabeled_train``,
    ``unlabeled_val``, ``unlabeled_test``, and optionally ``unlabeled_unlocated``
    (segments from image folders with no labeled siblings). Unlabeled splits
    yield samples whose ``target`` is the all-``IGNORE_LABEL_INDEX`` tensor.

    Differs from :func:`make_bih_data` only in defaults:

    - ``dataset_dir`` defaults to ``<root>/IRAP_Vietnam`` and ``metadata_dir`` to
      ``<dataset_dir>`` (Vietnam colocates data and metadata).
    - ``use_ncontext_filter=False``: the Vietnam release does not include the
      precomputed N-context pickles.
    - ``allow_missing_attributes=True``: five flow attributes (motorcycle /
      bicycle / pedestrian) are empty in every coding table — without this
      flag every segment would be dropped. Missing/unmappable codes are
      mapped to PyTorch's standard ``ignore_index = -1``.

    All other keyword arguments are forwarded to :func:`make_irap_data`.
    """
    if "metadata_dir" in kwargs:
        if kwargs.pop("metadata_dir") != dataset_dir:
            raise ValueError(
                "metadata_dir should not be passed explicitly or should be the same as dataset_dir"
            )

    if dataset_dir is None:
        dataset_dir = resolve_datasets_root() / "IRAP_Vietnam"

    data = make_irap_data(
        dataset_dir=dataset_dir,
        metadata_dir=dataset_dir,
        use_ncontext_filter=use_ncontext_filter,
        allow_missing_attributes=allow_missing_attributes,
        **kwargs,
    )

    if 'test' in data and len(data['test']) == 0:
        print("Warning: The 'test' split of the iRAP Vietnam dataset is empty and will be removed from the returned dataset dict.")
        del data['test']

    return data


# Registry of IRAP release presets. Single source of truth for the dataset-name
# strings used by CLI tools (vlm_inference, prepare_agent_tasks, ...).
IRAP_DATASET_FACTORIES = {
    "bih": make_bih_data,
    "vietnam": make_vietnam_data,
}


def make_irap_data_by_name(name: str, **kwargs):
    """Build an IRAP dataset dict by release name (``"bih"`` / ``"vietnam"``).

    Thin dispatch over :data:`IRAP_DATASET_FACTORIES`; forwards ``kwargs`` to the
    selected preset. Raises ``ValueError`` for an unknown name.
    """
    try:
        factory = IRAP_DATASET_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown IRAP dataset {name!r}. Choose from {sorted(IRAP_DATASET_FACTORIES)}."
        ) from None
    return factory(**kwargs)


def get_class_counts(metadata_dir: str | Path) -> tuple[int, ...]:
    """Number of classes per attribute, read directly from metadata.

    Lightweight: only loads ``attribute_metadata.json``. Useful in model
    factory expressions where constructing a dataset just to read
    ``info.class_counts`` would be wasteful.
    """
    ordered_attrs, attribute_value_to_irap = load_attribute_metadata(metadata_dir=metadata_dir)
    return tuple(len(attribute_value_to_irap[attr]) for attr in ordered_attrs)


def get_bih_class_counts(metadata_dir: str | Path | None = None) -> tuple[int, ...]:
    _, metadata_dir = resolve_irap_paths(metadata_dir=metadata_dir)
    return get_class_counts(metadata_dir)


def load_attribute_metadata(
    metadata_dir: str | Path,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Load IRAP attribute metadata and return attributes in canonical order.

    Args:
        metadata_dir: Metadata directory.

    Returns:
        ordered_attrs: Attribute names ordered by their index in the metadata.
        attribute_value_to_irap: Mapping attr -> {value -> irap_number}.
    """
    with open(metadata_dir / MetaFiles.ATTRIBUTE_METADATA, "r") as f:
        attr_meta = json.load(f)

    idx_to_attribute = {v: k for k, v in attr_meta["attribute_to_idx"].items()}
    ordered_attrs = [idx_to_attribute[i] for i in range(len(idx_to_attribute))]

    attribute_value_to_irap_number = attr_meta["attribute_value_to_irap_number"]
    return ordered_attrs, attribute_value_to_irap_number


class IRAPDataset(Dataset):
    """
    IRAP road sequence dataset.

    Expects a config dict with keys:
      - dataset_path (root for data files)
      - segment_id_to_data_paths_path (json mapping seg_id -> {rgb, depth})
      - splits_path (json with {train,val,test}: list[str])
      - road_id_to_segment_id_sequence_path (for context; optional)
      - context_offsets: list[int], e.g. [0, -1, -4]

    For each segment index, returns a dict with keys:
      - rgb: Tensor sequence of shape (S, C, H, W)
      - depth: optional tensor sequence (S, 1, H, W)
      - target: LongTensor with shape (A,) if labels provided via label_map
      - segment_id: str
    """

    # Canonical labeled subsets. The constructor accepts any key present in
    # `splits.json` (including `unlabeled_*` ones produced by the Vietnam prep
    # pipeline), but this tuple names the splits expected to carry labels.
    subsets = ("train", "val", "test")

    def __init__(
        self,
        root: T.Union[str, Path],
        subset: str = "train",
        *,
        metadata_dir: T.Union[str, Path, None] = None,
        context_offsets: T.Sequence[int] = (0, -1, -4),
        mean: T.Sequence[float] = RGB_MEAN,
        std: T.Sequence[float] = RGB_STD,
        transforms: T.Mapping[str, T.Callable] | None = None,
        ncontext_segment_id_subset: set[str] | None = None,
        allow_missing_attributes: bool = False,
    ) -> None:
        # unused attributes
        self.transforms = transforms or {}

        self.root = Path(root)
        self.metadata_dir = (
            Path(metadata_dir) if metadata_dir is not None
            else self.root.parent / (self.root.name + "_METADATA")
        )
        self.context_offsets = list(context_offsets)

        with open(self.metadata_dir / MetaFiles.SPLITS, "r") as f:
            all_splits = json.load(f)

        if subset not in all_splits:
            raise ValueError(
                f'Invalid subset "{subset}" for {type(self).__name__}. '
                f'Available in splits.json: {", ".join(sorted(all_splits))}.'
            )

        # Unlabeled splits have no entry in segment_id_to_road_data.json; force
        # allow_missing_attributes so segments are kept with all-IGNORE targets
        # instead of being dropped.
        if subset.startswith("unlabeled"):
            allow_missing_attributes = True

        with open(self.metadata_dir / MetaFiles.SEGMENT_ID_TO_DATA_PATHS, "r") as f:
            seg_to_paths_rel = json.load(f)
        self.seg_to_paths = {
            sid: {k: (None if v == "NONE" else (self.root / v)) for k, v in d.items()}
            for sid, d in tqdm(seg_to_paths_rel.items(), desc="Building seg_to_paths")
        }

        # Replicate DatasetWrapper.get_splits_and_contexts initial filtering:
        # filter all splits by segments that have data paths.
        # This must be done BEFORE context validation so that context IDs
        # are only drawn from segments with valid data.
        splits = {
            split_name: [seg_id for seg_id in segment_ids if seg_id in self.seg_to_paths]
            for split_name, segment_ids in all_splits.items()
        }

        # Valid segment IDs for subset (must exist in data paths)
        subset_ids = list(splits[subset])

        # Load attribute metadata and derive class indices
        ordered_attrs, attr_to_value_to_irap_number = load_attribute_metadata(metadata_dir=self.metadata_dir)
        # Build segment_id_to_labels FIRST (before context filtering) to match original order
        # This matches SeqEnhDatasetFromFeats which builds labels for all segments first
        # invert value->irap_number mapping
        attr_irap_to_value = {attr: {v: k for k, v in attr_to_value_to_irap_number[attr].items()} for attr in ordered_attrs}
        # enumerate new values per attribute to class indices
        attr_to_value_to_class_idx = {
            attr: {nv: i for i, nv in enumerate(attr_to_value_to_irap_number[attr].keys())}
            for attr in ordered_attrs
        }
        # Store attribute information directly from metadata computation
        class_counts = tuple(len(attr_to_value_to_irap_number[attr]) for attr in ordered_attrs)

        # Build labels for ALL segments in subset_ids (before context filtering)
        # This matches the original implementation which filters by labels first.
        # Additionally, we replicate DatasetWrapper._remove_filtered_out_segments:
        # segments for which the mapping changes the value (value != new_value)
        # are discarded entirely.
        with open(self.metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA, "r") as f:
            seg_to_road = json.load(f)
        # When `allow_missing_attributes` is True, missing or unmappable codes become
        # IGNORE_LABEL_INDEX (PyTorch's standard ignore_index for cross-entropy). This
        # is needed for datasets where some attribute columns are universally empty
        # (e.g. IRAP-Vietnam's flow attributes), so segments aren't all dropped.
        lm = {}
        for sid in tqdm(subset_ids, desc="Building label_map"):
            attrs_irap = seg_to_road.get(sid, {}).get("required_attributes", {})
            labels = []
            ok = True
            for attr in ordered_attrs:
                irap_code = attrs_irap.get(attr)
                if irap_code is None:
                    if allow_missing_attributes:
                        labels.append(IGNORE_LABEL_INDEX)
                        continue
                    ok = False
                    break
                value = attr_irap_to_value[attr].get(irap_code, None)
                if value is None:
                    if allow_missing_attributes:
                        labels.append(IGNORE_LABEL_INDEX)
                        continue
                    ok = False
                    break
                labels.append(attr_to_value_to_class_idx[attr][value])
            if ok:
                lm[sid] = labels
        self.segment_id_to_labels = lm

        # Filter subset_ids by labels FIRST (matches original: filter by labels, then context)
        subset_ids = [sid for sid in subset_ids if sid in self.segment_id_to_labels]

        # Load road sequence mapping BEFORE context filtering (needed for validation)
        # The file is optional: if it does not exist, we proceed without road sequences.
        try:
            with open(self.metadata_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, "r") as f:
                self.road_to_seq = json.load(f)
        except FileNotFoundError:
            self.road_to_seq = {}

        self.seq_index: dict[str, tuple[str, int]] = {}
        for road_id, seg_seq in self.road_to_seq.items():
            for i, sid in enumerate(seg_seq):
                self.seq_index[sid] = (road_id, i)

        # Use road-based sequence indexing to resolve context IDs, enforcing road boundaries.
        # Integer arithmetic on segment IDs is wrong when roads have adjacent IDs (e.g. Vietnam).
        # A context frame is valid iff its image exists on disk — not iff it appears in
        # some split — so check against seg_to_paths rather than the split union.
        self.segment_id_to_context_ids = {}
        valid_segment_ids = []
        for sid in subset_ids:
            if sid not in self.seq_index:
                continue
            road_id, pos = self.seq_index[sid]
            seq = self.road_to_seq[road_id]
            indices = [pos + offset for offset in self.context_offsets]
            if min(indices) < 0 or max(indices) >= len(seq):
                continue
            context_ids = tuple(seq[i] for i in indices)
            if all(cid in self.seg_to_paths for cid in context_ids):
                self.segment_id_to_context_ids[sid] = context_ids
                valid_segment_ids.append(sid)

        # Apply ncontext_segment_id_subset filter if provided (matches original behavior)
        # This filters splits to only include segment IDs in the provided set
        if ncontext_segment_id_subset is not None:
            valid_segment_ids = [sid for sid in valid_segment_ids if sid in ncontext_segment_id_subset]
            # Also filter context_ids mapping
            self.segment_id_to_context_ids = {
                sid: ctx_ids
                for sid, ctx_ids in self.segment_id_to_context_ids.items()
                if sid in ncontext_segment_id_subset
            }

        self.segment_ids = valid_segment_ids

        # Ensure consistency: all segment_ids must have labels (defensive check)
        missing_labels = [sid for sid in self.segment_ids if sid not in self.segment_id_to_labels]
        if missing_labels:
            raise ValueError(
                f"Internal error: Found {len(missing_labels)} segment IDs without labels after filtering. "
                f"This should not happen - filtering by labels occurs before context filtering. "
                f"First few missing: {missing_labels[:5]}"
            )
        super().__init__(
            subset=subset,
            info=dict(
                problem="multi_attribute_classification",
                class_counts=class_counts,
                pixel_stats=SimpleNamespace(mean=np.array(mean), std=np.array(std)),
                attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            ),
        )

    def __len__(self) -> int:
        return len(self.segment_ids)

    def _load_sequence(self, seq_ids: T.Sequence[str], kind: str) -> torch.Tensor:
        tfm = self.transforms.get(kind)
        frames: list[torch.Tensor] = []
        for sid in seq_ids:
            p = self.seg_to_paths[sid].get(kind)
            if p is None:
                continue
            arr = load_image_cv2(str(p))
            t = tfm(arr) if tfm is not None else hwc_to_chw_float_tensor(arr)
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(np.asarray(t))
            frames.append(t.float())
        if not frames:
            return torch.empty(0)
        return torch.stack(frames, dim=0)

    def get_example(self, idx: int) -> dict:
        sid = self.segment_ids[idx]

        # Prepare context ids from integer arithmetic (matches original).
        # All segment_ids are expected to have corresponding context ids; if not, this is an error.
        if sid not in self.segment_id_to_context_ids:
            raise KeyError(
                f"segment_id {sid} missing from segment_id_to_context_ids; dataset filtering should ensure consistency."
            )
        context_ids = list(self.segment_id_to_context_ids[sid])

        item = dict()
        item["rgb"] = self._load_sequence(context_ids, "rgb")
        if sid in self.segment_id_to_labels:
            item["target"] = torch.LongTensor(self.segment_id_to_labels[sid])
        item["segment_id"] = sid
        item["sequence_id"] = self.seq_index[sid][0]
        return item
