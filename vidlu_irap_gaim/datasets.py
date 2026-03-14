import json
import os
import pickle
import math
from pathlib import Path
import typing as T
import warnings

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torchvision.transforms import transforms as T_trans

from vidlu.data import Dataset, Record
from vidlu.data.datasets.datasets import _check_subset


RGB_MEAN: tuple[float, float, float] = (0.53354913, 0.52727484, 0.48752149)
RGB_STD: tuple[float, float, float] = (0.20401913, 0.20417478, 0.25402164)
INPUT_DIM_RGB: tuple[int, int, int] = (384, 288, 3)


def _load_image_cv2(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


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


class InferenceImageDataset(Dataset):
    """
    Dataset for running inference on arbitrary images with temporal context.

    Unlike BihSequence, this dataset:
    - Does NOT require labels (for pure inference without ground truth)
    - Uses alphabetical order of images to determine temporal context
    - Borrows attribute metadata from a reference dataset or explicit config

    Context sequence example:
        context_sequence=(0, -1, -4) means for image at index i, load images at i, i-1, i-4.
        Images without valid context (near the start) are excluded.

    Example usage:
        # From folder with context matching BihSequence
        ds = InferenceImageDataset.from_folder(
            "/path/to/images",
            reference_dataset=bih_test,
            context_sequence=(0, -1, -4),
        )
    """

    def __init__(
        self,
        image_paths: T.Sequence[Path],
        *,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        class_counts: tuple[int, ...],
        context_sequence: T.Sequence[int] = (0, -1, -4),
        mean: T.Sequence[float] = RGB_MEAN,
        std: T.Sequence[float] = RGB_STD,
        input_size: tuple[int, int] = (INPUT_DIM_RGB[0], INPUT_DIM_RGB[1]),  # (W, H)
    ):
        """
        Args:
            image_paths: Ordered sequence of image paths (alphabetical order assumed).
            attr_to_value_to_class_idx: Attribute metadata for decoding predictions.
            class_counts: Number of classes per attribute.
            context_sequence: Offsets for context frames, e.g., (0, -1, -4).
                The frames are ordered by offset (so (0, -1, -4) gives [img_0, img_-1, img_-4]).
            mean, std: Normalization stats (used by input adapter, stored in info).
            input_size: Target image size (W, H).
        """
        all_image_paths = [Path(p) for p in image_paths]
        self.context_sequence = tuple(context_sequence)
        self.input_size = input_size

        # Determine which images have valid context (all context indices in bounds)
        min_offset = min(self.context_sequence)
        max_offset = max(self.context_sequence)
        num_images = len(all_image_paths)

        # Valid indices: those where idx + min_offset >= 0 and idx + max_offset < num_images
        self.valid_indices: list[int] = []
        for idx in range(num_images):
            if idx + min_offset >= 0 and idx + max_offset < num_images:
                self.valid_indices.append(idx)

        self.all_image_paths = all_image_paths

        # Build transform: preserve aspect ratio (no stretching).
        #
        # We do a "resize-to-cover + center-crop" to match the fixed model input size while
        # avoiding geometric distortion:
        #   scale = max(target_w / w, target_h / h)
        #   resize(w*scale, h*scale) then center-crop to (target_h, target_w)
        self._to_tensor = T_trans.ToTensor()  # [0, 1]

        super().__init__(
            subset="inference",
            info=Record(
                problem="multi_attribute_classification",
                class_counts=class_counts,
                pixel_stats=Record(mean=np.array(mean), std=np.array(std)),
                attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            ),
        )

        print(
            f"[InferenceImageDataset] {len(all_image_paths)} images, "
            f"{len(self.valid_indices)} with valid context (sequence={self.context_sequence})"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def get_example(self, idx: int) -> Record:
        # Map dataset index to actual image index
        img_idx = self.valid_indices[idx]
        path = self.all_image_paths[img_idx]

        # Use keyword-only parameters (after *) to ensure positional_param_count returns 0
        # This prevents LazyItem from passing the record as an argument
        def load_rgb(
            *,
            img_idx=img_idx,
            context_sequence=self.context_sequence,
            all_paths=self.all_image_paths,
            input_size=self.input_size,
            to_tensor=self._to_tensor,
        ):
            """Lazy-load and preprocess RGB sequence."""
            target_w, target_h = input_size

            def preprocess_rgb_np(rgb_np: np.ndarray, img_path) -> torch.Tensor:
                from PIL import Image

                pil = Image.fromarray(rgb_np)
                w, h = pil.size
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid image size: {(w, h)} for {img_path}")

                scale = max(target_w / w, target_h / h)
                new_w = max(target_w, int(math.ceil(w * scale)))
                new_h = max(target_h, int(math.ceil(h * scale)))
                if (new_w, new_h) != (w, h):
                    pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                left = max(0, (new_w - target_w) // 2)
                top = max(0, (new_h - target_h) // 2)
                pil = pil.crop((left, top, left + target_w, top + target_h))

                return to_tensor(pil)

            frames = []
            for offset in context_sequence:
                ctx_idx = img_idx + offset
                ctx_path = all_paths[ctx_idx]
                img = _load_image_cv2(str(ctx_path))
                img_t = preprocess_rgb_np(img, ctx_path)
                frames.append(img_t)

            return torch.stack(frames, dim=0)

        return Record(
            rgb_=load_rgb,  # "_" suffix for lazy evaluation
            segment_id=path.stem,
            # No 'target' key - unlabeled
        )

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        reference_dataset: "BihSequence | None" = None,
        attr_to_value_to_class_idx: dict | None = None,
        class_counts: tuple[int, ...] | None = None,
        context_sequence: T.Sequence[int] = (0, -1, -4),
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        **kwargs,
    ) -> "InferenceImageDataset":
        """
        Create dataset from a folder of images.

        Images are sorted alphabetically to determine temporal order.
        Only images with valid context (all context indices in bounds) are included.

        Args:
            folder: Path to folder containing images.
            reference_dataset: A BihSequence to copy metadata from (alternative to explicit args).
            attr_to_value_to_class_idx: Explicit attribute metadata (if no reference_dataset).
            class_counts: Explicit class counts (if no reference_dataset).
            context_sequence: Offsets for context frames, e.g., (0, -1, -4).
            extensions: Image file extensions to include.
            **kwargs: Additional arguments passed to __init__.
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        # Collect image paths in alphabetical order
        image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ])
        if not image_paths:
            raise ValueError(f"No images with extensions {extensions} found in {folder}")

        # Get metadata
        if reference_dataset is not None:
            attr_to_value_to_class_idx = reference_dataset.info.attr_to_value_to_class_idx
            class_counts = reference_dataset.info.class_counts
        elif attr_to_value_to_class_idx is None or class_counts is None:
            raise ValueError(
                "Either reference_dataset or both attr_to_value_to_class_idx and class_counts must be provided"
            )

        return cls(
            image_paths,
            attr_to_value_to_class_idx=attr_to_value_to_class_idx,
            class_counts=class_counts,
            context_sequence=context_sequence,
            **kwargs,
        )
