from pathlib import Path
from typing import Protocol, Sequence

import json
import numpy as np
import torch

from irap_data import Dataset, IGNORE_LABEL_INDEX, load_attribute_metadata

from .feats import pack_features, read_attribute_slices


def _load_json(p: str | Path):
    with open(p, "r") as f:
        return json.load(f)


def resolve_attribute_index(attribute: int | str, ordered_attrs: list[str]) -> int:
    """The index of an attribute given by index or by name."""
    if isinstance(attribute, int):
        return attribute
    if attribute not in ordered_attrs:
        raise ValueError(f"Unknown attribute {attribute!r}. Available: {', '.join(ordered_attrs)}.")
    return ordered_attrs.index(attribute)


def _build_label_map(
    attribute_metadata_path: str | Path,
    segment_id_to_road_data_path: str | Path,
    attribute_value_mapping_path: str | Path | None,
    ordered_attrs: list[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Builds a mapping from segment ID to attribute class indices and counts classes per attribute.

    An attribute a segment has no (mappable) value for gets `IGNORE_LABEL_INDEX`.

    Args:
        attribute_metadata_path: Path to `attribute_metadata.json`.
        segment_id_to_road_data_path: Path to `segment_id_to_road_data.json`.
        attribute_value_mapping_path: Optional path to attribute value remapping JSON.
        ordered_attrs: Canonical list of attribute names.

    Returns:
        Tuple of `(segment_id_to_labels, attr_to_num_classes)`.
    """
    attr_meta = _load_json(attribute_metadata_path)
    value_to_irap = attr_meta["attribute_value_to_irap_number"]
    irap_to_value = {attr: {v: k for k, v in value_to_irap[attr].items()} for attr in ordered_attrs}
    if attribute_value_mapping_path:
        attribute_to_value_to_new_value = _load_json(attribute_value_mapping_path)
    else:
        attribute_to_value_to_new_value = {attr: {v: v for v in value_to_irap[attr].keys()} for attr in ordered_attrs}
    new_to_class = {
        attr: {nv: i for i, nv in enumerate(attribute_to_value_to_new_value[attr].values())} for attr in ordered_attrs
    }
    attr_to_num_classes = {attr: 1 + max(new_to_class[attr].values()) for attr in ordered_attrs}
    seg2road = _load_json(segment_id_to_road_data_path)
    seg_to_labels: dict[str, list[int]] = {}
    for seg_id, road_data in seg2road.items():
        attrs_irap = road_data.get("required_attributes", {})
        labels: list[int] = []
        for attr in ordered_attrs:
            irap = attrs_irap.get(attr)
            value = None if irap is None else irap_to_value[attr].get(irap)
            if value is None:
                labels.append(IGNORE_LABEL_INDEX)
                continue
            new_value = attribute_to_value_to_new_value[attr].get(value, value)
            labels.append(new_to_class[attr][new_value])
        seg_to_labels[seg_id] = labels
    return seg_to_labels, attr_to_num_classes


class DataSource(Protocol):
    """
    Abstract protocol for data sources.
    Retrieves data for a sequence of context IDs.
    """

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor: ...


class PackedSegmentArray:
    """One row per segment, memory-mapped from a directory's packed store.

    The store is memory-mapped, so several dataloader workers (and the datasets of all
    attributes) share one copy through the page cache instead of reopening per-segment files
    for every example.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        array_path, index_path = pack_features(self.directory)
        self.array = np.load(array_path, mmap_mode="r")
        with open(index_path) as f:
            self.segment_id_to_row = json.load(f)

    @property
    def dim(self) -> int:
        return int(self.array.shape[1])

    def has(self, segment_id: str) -> bool:
        return segment_id in self.segment_id_to_row

    def rows(self, segment_ids: list[str], columns: slice | None = None) -> np.ndarray:
        """The rows of `segment_ids`, in order; `columns` selects part of each row."""
        indices = [self.segment_id_to_row[sid] for sid in segment_ids]
        return self.array[indices] if columns is None else self.array[indices, columns]


class FeatDataSource:
    """Feature vectors of the base model, from a feature directory's packed store."""

    def __init__(self, store: PackedSegmentArray):
        self.store = store

    @property
    def feat_dim(self) -> int:
        return self.store.dim

    def has_features(self, segment_id: str) -> bool:
        return self.store.has(segment_id)

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor:
        # Expected shape: (S, D)
        return torch.from_numpy(self.store.rows(context_ids).astype(np.float32))


class LogitDataSource:
    """The base model's logits for one attribute, from a logit directory's packed store.

    The counterpart of the original implementation's `LogitDataAccess`: what the
    sequence carries at each position is the per-segment classifier's *output*, which
    is what sequential enhancement refines. Ground truth enters only as the target.
    """

    def __init__(self, store: PackedSegmentArray, columns: slice):
        self.store = store
        self.columns = columns

    @property
    def num_classes(self) -> int:
        return self.columns.stop - self.columns.start

    def has_features(self, segment_id: str) -> bool:
        return self.store.has(segment_id)

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor:
        # Expected shape: (S, n_classes)
        return torch.from_numpy(self.store.rows(context_ids, self.columns).astype(np.float32))


class LabelDataSource:
    """The base model's predicted class for one attribute, per segment of the sequence.

    The original implementation's `LabelDataAccess`, which reads `y_pred`. Derived from
    the stored logits rather than stored separately, so the two cannot disagree; and
    being a prediction it is always in `[0, num_classes)`, so the embedding needs no row
    for a missing value.
    """

    def __init__(self, store: PackedSegmentArray, columns: slice):
        self.logits = LogitDataSource(store, columns)

    @property
    def num_classes(self) -> int:
        return self.logits.num_classes

    def has_features(self, segment_id: str) -> bool:
        return self.logits.has_features(segment_id)

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor:
        # (S, 1), matching the (B, S, D) shape the encoders concatenate along.
        return self.logits.get_sequence(context_ids).argmax(dim=-1, keepdim=True)


def validate_context_offsets(context_offsets: Sequence[int]) -> list[int]:
    """Checks a smoothing window and returns it as a list.

    The window must contain the target (offset 0) and run forwards in time. The order
    is not cosmetic -- an LSTM reads the sequence in the order given -- so a
    non-increasing tuple is rejected rather than sorted: silently reinterpreting a
    configured `(0, -1, -4)` as `(-4, -1, 0)` would change what an existing command does
    without saying so.
    """
    offsets = list(context_offsets)
    if 0 not in offsets:
        raise ValueError(
            f"context_offsets={tuple(offsets)} does not contain 0, so the segment being"
            f" classified is not in its own context window.")
    if any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError(
            f"context_offsets={tuple(offsets)} is not strictly increasing. The sequence is"
            f" fed to the model in this order, so it has to run forwards in time; write it"
            f" as {tuple(sorted(offsets))}.")
    return offsets


class SeqEnhDataset(Dataset):
    """Sequential enhancement dataset for temporal smoothing over segment feature sequences.

    Returns samples containing:
      - `data`: Input sequence tensor of shape `(sequence_length, feature_dim)` or dict of sources.
      - `target`: Ground-truth scalar class label for the target segment.
      - `segment_id`: Target segment identifier string.
    """

    subsets = ("train", "val", "test")

    def __init__(
        self,
        subset: str,
        *,
        data_sources: dict[str, DataSource],
        segment_id_list: list[str],
        seq_index: dict[str, tuple[str, int]],
        road_to_seq: dict[str, list[str]],
        context_offsets: Sequence[int],
        target_label_map: dict[str, list[int]],
        target_attribute_index: int,
        class_count: int,
        feat_dim: int | None = None,
    ) -> None:
        context_offsets = validate_context_offsets(context_offsets)
        super().__init__(subset=subset,
                         info=dict(problem="classification", class_count=class_count,
                                   feat_dim=feat_dim,
                                   # Where the classified segment sits in the sequence. The
                                   # model reads its local context there, so it is published
                                   # rather than left to be guessed from the sequence length.
                                   target_index=context_offsets.index(0)))
        self.data_sources = data_sources
        self.segment_ids = segment_id_list
        # Optimization: convert lists to tuples if static
        self.seq_index = seq_index
        self.road_to_seq = road_to_seq
        self.context_offsets = list(context_offsets)
        self.target_label_map = target_label_map
        self.target_attribute_index = target_attribute_index

    def __len__(self) -> int:
        return len(self.segment_ids)

    def get_example(self, idx: int) -> dict:
        sid = self.segment_ids[idx]
        road_id, pos = self.seq_index[sid]
        seq = self.road_to_seq[road_id]

        # Checked here, not only in `make_seq_enh_data`: a negative `pos + off` is a valid
        # Python index, so directly constructed datasets would silently wrap to the end of the
        # road instead of erroring.
        positions = [pos + off for off in self.context_offsets]
        if min(positions) < 0 or max(positions) >= len(seq):
            raise IndexError(
                f"Context {positions} of segment {sid!r} (position {pos} of road {road_id!r},"
                f" length {len(seq)}) falls outside the road sequence.")
        ids = [seq[p] for p in positions]

        input_data = {name: source.get_sequence(ids) for name, source in self.data_sources.items()}
        if len(input_data) == 1:
            (input_data,) = input_data.values()

        y = int(self.target_label_map[sid][self.target_attribute_index])

        return dict(data=input_data, target=torch.tensor(y, dtype=torch.int64), segment_id=sid)


DEFAULT_CONTEXT_OFFSETS = tuple(range(-10, 11))


def make_seq_enh_data(
    *,
    feat_dir: str | Path,
    attribute: int | str,
    context_offsets: Sequence[int] = DEFAULT_CONTEXT_OFFSETS,
    metadata_dir: str | Path | None = None,
    irap_home: str | Path | None = None,
    attribute_value_mapping_path: str | Path | None = None,
    input_types: Sequence[str] = ("feats",),
    logit_dir: str | Path | None = None,
):
    """Constructs sequential enhancement train, val, and test datasets for an attribute.

    Every input the sequence carries is an *output* of the base per-segment classifier --
    its features, its logits, or the class it predicted -- which is what this stage
    refines. Ground truth enters only as the target, for the classified segment.

    Args:
        feat_dir: Directory containing extracted feature `.npy` files.
        attribute: Attribute index (int) or attribute name (str).
        context_offsets: Offsets, in segments along the road, of the smoothing window
            around the classified segment. Must contain 0 and be strictly increasing;
            defaults to a symmetric window of radius 10. Unrelated to the frame context
            the base model itself sees: features are per segment.
        metadata_dir: Explicit path to metadata directory.
        irap_home: Optional IRAP home directory for locating metadata.
        attribute_value_mapping_path: Optional path to attribute value remapping JSON.
        input_types: Input modalities to load, any of `('feats', 'logits', 'labels')`.
            More than one makes each example's `data` a mapping keyed by input type; a
            single one yields a bare tensor.
        logit_dir: Directory of extracted logits, required by the 'logits' and 'labels'
            inputs. Written by `extract_features(..., logit_dir=...)`.

    Returns:
        LazyDict mapping split names ('train', 'val', 'test') to `SeqEnhDataset` instances.
    """
    context_offsets = validate_context_offsets(context_offsets)
    known_input_types = ("feats", "logits", "labels")
    if unknown := [t for t in input_types if t not in known_input_types]:
        raise ValueError(f"Unknown input_types {unknown}; known: {list(known_input_types)}.")
    needs_logits = any(t in input_types for t in ("logits", "labels"))
    if needs_logits and logit_dir is None:
        raise ValueError(
            f"input_types={tuple(input_types)} needs the base model's logits, so `logit_dir`"
            f" is required. It is written by `extract_features(..., logit_dir=...)`.")

    # --- 1. Resolve Paths & Metadata ---
    if metadata_dir is None:
        import os

        home = Path(irap_home) if irap_home is not None else Path(os.environ.get("IRAP_HOME", ""))
        if home == Path(""):
            raise RuntimeError(
                "make_seq_enh_data: metadata_dir is not set and neither is irap_home or the"
                " IRAP_HOME environment variable. Pass metadata_dir explicitly (for releases"
                " other than IRAP-BiH it differs from the default"
                " $IRAP_HOME/IRAP_BIH_METADATA; IRAP-Vietnam colocates metadata with the"
                " dataset directory).")
        metadata_dir = home / "IRAP_BIH_METADATA"
    metadata_dir = Path(metadata_dir)

    # --- 2. Load Common Metadata ---
    road_to_seq = _load_json(metadata_dir / "road_id_to_segment_id_sequence.json")

    ordered_attrs, _ = load_attribute_metadata(metadata_dir)

    attribute_index = resolve_attribute_index(attribute, ordered_attrs)
    attribute_name = ordered_attrs[attribute_index]

    # --- 3. Build Label Map (for Targets & Label Input) ---
    seg_to_labels, attr_to_num_classes = _build_label_map(
        metadata_dir / "attribute_metadata.json",
        metadata_dir / "segment_id_to_road_data.json",
        attribute_value_mapping_path,
        ordered_attrs,
    )

    # --- 4. Build Sequence Index ---
    seq_index = {}
    for road_id, seg_seq in road_to_seq.items():
        for i, sid in enumerate(seg_seq):
            seq_index[sid] = (road_id, i)

    # --- 5. Prepare Subsets ---
    splits = _load_json(metadata_dir / "splits.json")
    datasets = {}

    # One source of each kind for all subsets: each memory-maps a packed store, and the
    # subsets differ only in which segments they select.
    data_sources: dict[str, DataSource] = {}
    if "feats" in input_types:
        data_sources["feats"] = FeatDataSource(PackedSegmentArray(feat_dir))
    if needs_logits:
        logit_store = PackedSegmentArray(logit_dir)
        columns = slice(*read_attribute_slices(logit_dir)[attribute_index])
        if "logits" in input_types:
            data_sources["logits"] = LogitDataSource(logit_store, columns)
        if "labels" in input_types:
            data_sources["labels"] = LabelDataSource(logit_store, columns)
    feat_dim = None if "feats" not in data_sources else data_sources["feats"].feat_dim

    class_count = attr_to_num_classes[attribute_name]
    for name in ("logits", "labels"):
        # The head that produced the logits and the label map that produces the target have
        # to describe the same class set; otherwise the inputs and the target for this
        # attribute mean different things and nothing downstream would notice.
        if name in data_sources and data_sources[name].num_classes != class_count:
            raise RuntimeError(
                f"The base model's head for {attribute_name!r} has"
                f" {data_sources[name].num_classes} classes, but the metadata in"
                f" {metadata_dir} gives it {class_count}. The logits in {Path(logit_dir)} were"
                f" extracted from a model built for different attribute metadata.")

    for subset in ["train", "val", "test"]:
        valid_ids = []
        # A window whose segments span splits would let a val/test segment be refined using
        # the base model's predictions on its *training* neighbours, which are optimistically
        # accurate. The original implementation requires one split per window; so does this.
        split_ids = set(splits[subset])
        # Segments are dropped silently one by one, so the reasons are counted and
        # reported when a subset ends up empty.
        num_dropped = dict(no_road_data=0, target_attribute_unlabeled=0,
                           not_in_road_sequence=0, context_out_of_range=0,
                           context_outside_split=0, missing_features=0)
        for sid in splits[subset]:
            if sid not in seg_to_labels:
                num_dropped["no_road_data"] += 1
                continue
            if seg_to_labels[sid][attribute_index] == IGNORE_LABEL_INDEX:
                num_dropped["target_attribute_unlabeled"] += 1
                continue
            if sid not in seq_index:
                num_dropped["not_in_road_sequence"] += 1
                continue
            road_id, idx = seq_index[sid]
            seq = road_to_seq[road_id]
            indices = [idx + off for off in context_offsets]
            if min(indices) < 0 or max(indices) >= len(seq):
                num_dropped["context_out_of_range"] += 1
                continue

            context_ids = [seq[i] for i in indices]
            if not all(cid in split_ids for cid in context_ids):
                num_dropped["context_outside_split"] += 1
                continue

            if not all(source.has_features(cid)
                       for source in data_sources.values() for cid in context_ids):
                num_dropped["missing_features"] += 1
                continue

            valid_ids.append(sid)

        if len(valid_ids) == 0 and len(splits[subset]) > 0:
            raise RuntimeError(
                f"make_seq_enh_data: all {len(splits[subset])} segments of subset '{subset}' were"
                f" dropped for attribute {attribute_name!r} (index {attribute_index}). Dropped per"
                f" reason: " + ", ".join(f"{k}={v}" for k, v in num_dropped.items())
                + f". metadata_dir={metadata_dir}, feat_dir={Path(feat_dir)},"
                f" context_offsets={tuple(context_offsets)}.")

        # The same source objects for every subset; the subsets differ only in `valid_ids`.
        datasets[subset] = SeqEnhDataset(
            subset=subset,
            data_sources=data_sources,
            segment_id_list=valid_ids,
            seq_index=seq_index,
            road_to_seq=road_to_seq,
            context_offsets=context_offsets,
            target_label_map=seg_to_labels,
            target_attribute_index=attribute_index,
            class_count=class_count,
            feat_dim=feat_dim,
        )

    return datasets
