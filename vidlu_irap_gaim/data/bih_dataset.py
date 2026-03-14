import json
import os
import pickle
import warnings
from pathlib import Path
import typing as T

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import transforms as T_trans

from vidlu.data import Dataset, Record
from vidlu.data.datasets.datasets import _check_subset

from .constants import RGB_MEAN, RGB_STD, INPUT_DIM_RGB, _load_image_cv2


def resolve_irap_paths(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve IRAP dataset and metadata directories.

    Args:
        dataset_dir: Optional explicit dataset directory (overrides irap_home-derived path).
        metadata_dir: Optional explicit metadata directory (overrides irap_home-derived path).

    Returns:
        Tuple of (dataset_dir, metadata_dir) as Path objects.

    Raises:
        RuntimeError: If irap_home cannot be resolved.
    """
    env_home = None
    if metadata_dir is None or dataset_dir is None:
        env_home = os.environ.get("IRAP_HOME", None)
        if env_home is None:
            raise RuntimeError("Cannot resolve irap_home: provide irap_home parameter or set IRAP_HOME env var.")
        irap_home = Path(env_home)
    md_dir = irap_home / "IRAP_BIH_METADATA" if env_home is not None else Path(metadata_dir)
    ds_dir = irap_home / "IRAP_BIH" if env_home is not None else Path(dataset_dir)
    return ds_dir, md_dir


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


def make_bih_data(
    *,
    dataset_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    context_sequence: T.Sequence[int] = (0, -1, -4),
    data_types: T.Sequence[str] = ("rgb",),
    mean: T.Sequence[float] = RGB_MEAN,
    std: T.Sequence[float] = RGB_STD,
    input_dim_rgb: T.Sequence[int] = INPUT_DIM_RGB,
    transforms: T.Mapping[str, T.Callable] | None = None,
    label_map: T.Mapping[str, T.Sequence[int]] | None = None,
    ncontext_segment_id_subset: set[str] | None = None,
    use_ncontext_filter: bool = True,
    seg_to_res_path: str | Path | None = None,
):
    """Build BiH datasets with IRAP GAIM configuration.

    Args:
        dataset_dir: Optional override for IRAP_BIH directory.
        metadata_dir: Optional override for IRAP_BIH_METADATA directory.
        context_sequence: Offsets for context frames, e.g., (0, -1, -4).
        data_types: Which data types to load, e.g., ("rgb",).
        mean: RGB channel means for normalization.
        std: RGB channel stds for normalization.
        input_dim_rgb: Target image dimensions (W, H, C).
        transforms: Custom transforms dict.
        label_map: Custom label mapping.
        ncontext_segment_id_subset: Explicit set of segment IDs to include.
        use_ncontext_filter: If True (default), apply N-context filtering using pickle
            files at seg_to_res_path (or metadata_dir/seg_to_res if seg_to_res_path is None).
            Set to False to skip N-context filtering entirely.
        seg_to_res_path: Path to directory containing precomputed result pickle files
            ({train,val,test}.pickle). If None and use_ncontext_filter=True, uses
            metadata_dir/seg_to_res. Only used if use_ncontext_filter=True.
    """
    # Resolve paths once at the top level (both derived from irap_home)
    ds_dir, md_dir = resolve_irap_paths(dataset_dir=dataset_dir, metadata_dir=metadata_dir)

    # Load segment ID filter from precomputed results if requested
    if use_ncontext_filter and ncontext_segment_id_subset is None:
        if seg_to_res_path is None:
            seg_to_res_path = md_dir / "seg_to_res"
        road_seq_path = md_dir / "road_id_to_segment_id_sequence.json"
        ncontext_segment_id_subset = load_ncontext_segment_ids(seg_to_res_path, road_seq_path)

    if transforms is None:
        # Build default transforms similar to original
        def build_rgb_transform(train: bool):
            # Photometric jittering is handled in TrainerConfig; keep loading deterministic here.
            ops = [
                T_trans.ToPILImage(),
                T_trans.CenterCrop((int(input_dim_rgb[1]), int(input_dim_rgb[0]))),
                T_trans.ToTensor(),  # leave in [0,1]; normalization handled by input adapter
            ]
            return T_trans.Compose(ops)

        def build_depth_transform():
            return T_trans.Compose([T_trans.ToPILImage(), T_trans.ToTensor()])

        transforms = dict(
            train=dict(rgb=build_rgb_transform(True), depth=build_depth_transform()),
            val=dict(rgb=build_rgb_transform(False), depth=build_depth_transform()),
            test=dict(rgb=build_rgb_transform(False), depth=build_depth_transform()),
        )

    ds_kwargs = dict(
        context_sequence=tuple(context_sequence),
        data_types=tuple(str(x) for x in data_types),
        mean=tuple(float(x) for x in mean),
        std=tuple(float(x) for x in std),
        label_map=label_map,
        ncontext_segment_id_subset=ncontext_segment_id_subset,
    )

    return {
        split: BihSequence(
            ds_dir,
            split,
            transforms=transforms.get(split) if isinstance(transforms, dict) else transforms,
            **ds_kwargs,
        )
        for split in ["train", "val", "test"]
    }


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
    with open(metadata_dir / "attribute_metadata.json", "r") as f:
        attr_meta = json.load(f)

    idx_to_attribute = {v: k for k, v in attr_meta["attribute_to_idx"].items()}
    ordered_attrs = [idx_to_attribute[i] for i in range(len(idx_to_attribute))]

    attribute_value_to_irap_number = attr_meta["attribute_value_to_irap_number"]
    return ordered_attrs, attribute_value_to_irap_number


def get_class_counts(
    metadata_dir: str | Path = None,
) -> tuple[int, ...]:
    """Get the number of classes for each attribute.

    Args:
        metadata_dir: Metadata directory.

    Returns:
        Tuple of class counts, one per attribute.
    """
    warnings.warn("get_class_counts is deprecated. Use Dataset.info.attr_to_class_count instead")
    _, metadata_dir = resolve_irap_paths(metadata_dir=metadata_dir)
    ordered_attrs, attribute_value_to_irap = load_attribute_metadata(metadata_dir=metadata_dir)
    return tuple(len(attribute_value_to_irap[attr]) for attr in ordered_attrs)


class BihSequence(Dataset):
    """
    Minimal reimplementation of IRAP GAIM sequence dataset for ViDLU.

    Expects a config dict with keys:
      - dataset_path (root for data files)
      - segment_id_to_data_paths_path (json mapping seg_id -> {rgb, depth})
      - splits_path (json with {train,val,test}: list[str])
      - road_id_to_segment_id_sequence_path (for context; optional)
      - context_sequence: list[int], e.g. [0, -1, -4]
      - data_types: ["rgb", "depth"] subset supported; default ["rgb"]

    For each segment index, returns a Record with keys:
      - rgb: Tensor sequence of shape (S, C, H, W)
      - depth: optional tensor sequence (S, 1, H, W)
      - target: LongTensor with shape (A,) if labels provided via label_map
      - segment_id: str
    """

    subsets = ("train", "val", "test")

    def __init__(
        self,
        root: T.Union[str, Path],
        subset: str = "train",
        *,
        context_sequence: T.Sequence[int] = (0, -1, -4),
        data_types: T.Sequence[str] = ("rgb",),
        mean: T.Sequence[float] = (0.53354913, 0.52727484, 0.48752149),
        std: T.Sequence[float] = (0.20401913, 0.20417478, 0.25402164),
        transforms: T.Mapping[str, T.Callable] | None = None,
        ncontext_segment_id_subset: set[str] | None = None,
    ) -> None:
        _check_subset(self.__class__, subset)
        # unused attributes
        self.transforms = transforms or {}

        self.root = Path(root)
        self.metadata_dir = self.root.parent / (self.root.name + "_METADATA")
        self.context_sequence = list(context_sequence)
        self.data_types = set(data_types)

        with open(self.metadata_dir / "splits.json", "r") as f:
            all_splits = json.load(f)
        with open(self.metadata_dir / "segment_id_to_data_paths_rel.json", "r") as f:
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
        attribute_names = list(ordered_attrs)
        class_counts = tuple(len(attr_to_value_to_irap_number[attr]) for attr in ordered_attrs)

        # Build labels for ALL segments in subset_ids (before context filtering)
        # This matches the original implementation which filters by labels first.
        # Additionally, we replicate DatasetWrapper._remove_filtered_out_segments:
        # segments for which the mapping changes the value (value != new_value)
        # are discarded entirely.
        with open(self.metadata_dir / "segment_id_to_road_data.json", "r") as f:
            seg_to_road = json.load(f)
        lm = {}
        for sid in tqdm(subset_ids, desc="Building label_map"):
            attrs_irap = seg_to_road.get(sid, {}).get("required_attributes", {})
            labels = []
            ok = True
            for attr in ordered_attrs:
                irap_code = attrs_irap.get(attr)
                if irap_code is None:
                    ok = False
                    break
                # Canonical value from IRAP code
                value = attr_irap_to_value[attr].get(irap_code, None)
                if value is None:
                    ok = False
                    break
                # If mapping changes the value, drop the segment (matches original)
                labels.append(attr_to_value_to_class_idx[attr][value])
            if ok:
                lm[sid] = labels
        self.segment_id_to_labels = lm

        # Filter subset_ids by labels FIRST (matches original: filter by labels, then context)
        subset_ids = [sid for sid in subset_ids if sid in self.segment_id_to_labels]

        # Load road sequence mapping BEFORE context filtering (needed for validation)
        # The file is optional: if it does not exist, we proceed without road sequences.
        try:
            with open(self.metadata_dir / "road_id_to_segment_id_sequence.json", "r") as f:
                self.road_to_seq = json.load(f)
        except FileNotFoundError:
            self.road_to_seq = {}

        self.seq_index: dict[str, tuple[str, int]] = {}
        for road_id, seg_seq in self.road_to_seq.items():
            for i, sid in enumerate(seg_seq):
                self.seq_index[sid] = (road_id, i)

        # Build contexts using integer offsets like the original implementation
        # The original uses integer arithmetic on segment IDs for context windows.
        # Context validity is computed over segments that *have data paths*.
        # So we build `all_segment_ids_int` from the filtered `splits` dict.
        all_splits_all_ids = []
        for split_name, segment_ids in tqdm(splits.items(), desc="Building all_segment_ids_int"):
            all_splits_all_ids.extend(segment_ids)
        all_segment_ids_int = set(map(int, all_splits_all_ids))

        # Filter to segments that have valid context when using integer offsets
        # The original requires segments to be in road sequences (seq_index) for context validation
        # This matches SeqEnhDatasetFromFeats which checks `if sid not in seq_index: continue`
        self.segment_id_to_context_ids = {}
        valid_segment_ids = []
        for sid in subset_ids:
            # Check if segment is in road sequences (matches original behavior)
            if sid not in self.seq_index:
                continue

            sid_int = int(sid)
            context_ids_int = [sid_int + offset for offset in self.context_sequence]
            context_ids_str = tuple(map(str, context_ids_int))
            # Check that all context IDs exist in splits (original only checks existence, not labels)
            if all(cont_id in all_segment_ids_int for cont_id in context_ids_int):
                self.segment_id_to_context_ids[sid] = context_ids_str
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
            info=Record(
                problem="multi_attribute_classification",
                class_counts=class_counts,
                pixel_stats=Record(mean=np.array(mean), std=np.array(std)),
                attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            ),
        )

    def __len__(self) -> int:
        return len(self.segment_ids)

    def _load_sequence(self, seq_ids: T.Sequence[str], kind: str) -> np.ndarray:
        frames = []
        tfm = self.transforms.get(kind)
        for sid in seq_ids:
            p = self.seg_to_paths[sid].get(kind)
            if p is None:
                continue
            arr = _load_image_cv2(str(p))
            if tfm is not None:
                arr = tfm(arr)
            # Convert to CHW float32 numpy in [0,1]
            if hasattr(arr, "numpy") and isinstance(arr, torch.Tensor):
                a = arr.detach().cpu().numpy()
            elif isinstance(arr, np.ndarray):
                # assume HWC uint8 -> convert to float32 CHW
                if arr.ndim == 3 and arr.shape[2] in (1, 3):
                    a = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
                else:
                    a = arr.astype(np.float32)
            else:
                # PIL.Image
                a = np.asarray(arr).transpose(2, 0, 1).astype(np.float32) / 255.0
            frames.append(a)
        if not frames:
            return np.zeros((0,))
        x = np.stack(frames, axis=0)
        return x

    def get_example(self, idx: int) -> Record:
        sid = self.segment_ids[idx]

        # Prepare context ids from integer arithmetic (matches original).
        # All segment_ids are expected to have corresponding context ids; if not, this is an error.
        if sid not in self.segment_id_to_context_ids:
            raise KeyError(
                f"segment_id {sid} missing from segment_id_to_context_ids; dataset filtering should ensure consistency."
            )
        context_ids = list(self.segment_id_to_context_ids[sid])

        items = []

        # RGB sequence as a tensor (lazy-loaded to avoid IO when only labels are needed)
        if "rgb" in self.data_types:
            # Use keyword-only parameter (after *) to ensure positional_param_count returns 0
            # This prevents LazyItem from passing the record as an argument
            def load_rgb(*, ctx_ids=context_ids):
                seq = self._load_sequence(ctx_ids, "rgb")
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).float()
                return seq
            items.append(("rgb_", load_rgb))  # "_" suffix for lazy evaluation

        # Labels are small, can be loaded directly – this will be `y`
        if sid in self.segment_id_to_labels:
            items.append(("target", torch.LongTensor(self.segment_id_to_labels[sid])))

        # Keep segment_id for bookkeeping/metrics
        items.append(("segment_id", sid))
        return Record(dict(items))
