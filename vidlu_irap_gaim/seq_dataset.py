from __future__ import annotations

from pathlib import Path
from typing import Protocol, Any, Sequence

import json
import numpy as np
import torch

from vidlu.data import Dataset, Record


def _load_json(p: str | Path):
    with open(p, "r") as f:
        return json.load(f)


def _build_label_map(
    attribute_metadata_path: str | Path,
    segment_id_to_road_data_path: str | Path,
    attribute_value_mapping_path: str | Path | None,
    ordered_attrs: list[str],
) -> dict[str, list[int]]:
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
    seg2road = _load_json(segment_id_to_road_data_path)
    seg_to_labels: dict[str, list[int]] = {}
    for seg_id, road_data in seg2road.items():
        attrs_irap = road_data.get("required_attributes", {})
        labels: list[int] = []
        ok = True
        for attr in ordered_attrs:
            irap = attrs_irap.get(attr)
            if irap is None:
                ok = False
                break
            value = irap_to_value[attr].get(irap)
            if value is None:
                ok = False
                break
            new_value = attribute_to_value_to_new_value[attr].get(value, value)
            labels.append(new_to_class[attr][new_value])
        if ok:
            seg_to_labels[seg_id] = labels
    return seg_to_labels


class DataSource(Protocol):
    """
    Abstract protocol for data sources.
    Retrieves data for a sequence of context IDs.
    """

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor: ...


class FeatDataSource:
    """
    Loads feature vectors from .npy files.
    """

    def __init__(self, feat_dir: str | Path):
        self.feat_dir = Path(feat_dir)

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor:
        # Load feats and flatten per frame, then stack
        # Expected shape: (S, D)
        frames = [np.load(self.feat_dir / f"{cid}.npy").reshape(-1).astype(np.float32) for cid in context_ids]
        return torch.from_numpy(np.stack(frames, axis=0))


class LabelDataSource:
    """
    Loads label indices from a label map.
    """

    def __init__(self, segment_id_to_labels: dict[str, list[int]], attribute_index: int):
        self.segment_id_to_labels = segment_id_to_labels
        self.attribute_index = attribute_index

    def get_sequence(self, context_ids: list[str]) -> torch.Tensor:
        # Expected shape: (S,) or (S, 1) depending on model expectation
        # Here we return (S, 1) to match (B, S, D) common shape
        labels = [self.segment_id_to_labels[cid][self.attribute_index] for cid in context_ids]
        return torch.tensor(labels, dtype=torch.long).unsqueeze(-1)


# TODO: Implement LogitDataSource if needed (requires pickle loading logic from old codebase)


class SeqEnhDataset(Dataset):
    """
    Modular Sequential Enhancement Dataset.
    Configured with a dictionary of DataSources.

    Returns a record with:
      - data: dict[str, Tensor] (keys matching data_sources)
      - target: Tensor (scalar label for the central segment)
      - segment_id: str
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
        context_sequence: Sequence[int],
        target_label_map: dict[str, list[int]],
        target_attribute_index: int,
    ) -> None:
        super().__init__(subset=subset, info=Record(problem="classification", class_count=0))
        self.data_sources = data_sources
        self.segment_ids = segment_id_list
        # Optimization: convert lists to tuples if static
        self.seq_index = seq_index
        self.road_to_seq = road_to_seq
        self.context_sequence = list(context_sequence)
        self.target_label_map = target_label_map
        self.target_attribute_index = target_attribute_index

    def __len__(self) -> int:
        return len(self.segment_ids)

    def get_example(self, idx: int) -> Record:
        sid = self.segment_ids[idx]
        road_id, pos = self.seq_index[sid]
        seq = self.road_to_seq[road_id]

        # Calculate context IDs
        ids = [seq[pos + off] for off in self.context_sequence]

        input_data = {}
        for name, source in self.data_sources.items():
            input_data[name] = source.get_sequence(ids)

        y = int(self.target_label_map[sid][self.target_attribute_index])

        return Record(data=input_data, target=torch.tensor(y, dtype=torch.int64), segment_id=sid)


def make_seq_enh_data(
    *,
    feat_dir: str | Path,
    attribute: int | str,
    context_sequence: Sequence[int] = (0, -1, -4),
    metadata_dir: str | Path | None = None,
    irap_home: str | Path | None = None,
    attribute_value_mapping_path: str | Path | None = None,
    # New args for flexibility
    input_types: Sequence[str] = ("feats",),  # 'feats', 'labels'
):
    """
    Factory function to create SeqEnhDataset instances (train/val/test).
    Maintains backward compatibility for 'feats' input while enabling new types.
    """

    # --- 1. Resolve Paths & Metadata ---
    if metadata_dir is None:
        import os

        home = Path(irap_home) if irap_home is not None else Path(os.environ.get("IRAP_HOME", ""))
        if home == Path(""):
            # Fallback for when basic info is missing, though usually fatal
            raise RuntimeError("IRAP_HOME/metadata_dir not set.")
        metadata_dir = home / "IRAP_BIH_METADATA"
    metadata_dir = Path(metadata_dir)

    # --- 2. Load Common Metadata ---
    road_to_seq = _load_json(metadata_dir / "road_id_to_segment_id_sequence.json")
    attr_meta = _load_json(metadata_dir / "attribute_metadata.json")

    idx_to_attribute = {v: k for k, v in attr_meta["attribute_to_idx"].items()}
    ordered_attrs = [idx_to_attribute[i] for i in range(len(idx_to_attribute))]

    if isinstance(attribute, int):
        attribute_index = int(attribute)
        attribute_name = ordered_attrs[attribute_index]
    else:
        attribute_name = str(attribute)
        attribute_index = ordered_attrs.index(attribute_name)

    # --- 3. Build Label Map (for Targets & Label Input) ---
    seg_to_labels = _build_label_map(
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

    for subset in ["train", "val", "test"]:
        # valid ids must have labels
        candidate_ids = [sid for sid in splits[subset] if sid in seg_to_labels]

        valid_ids = []
        for sid in candidate_ids:
            if sid not in seq_index:
                continue
            road_id, idx = seq_index[sid]
            seq = road_to_seq[road_id]
            indices = [idx + off for off in context_sequence]
            if min(indices) < 0 or max(indices) >= len(seq):
                continue

            # Check file existence for features if 'feats' is required
            if "feats" in input_types:
                context_sids = [seq[i] for i in indices]
                if not all((Path(feat_dir) / f"{cid}.npy").exists() for cid in context_sids):
                    continue

            valid_ids.append(sid)

        # --- 6. Instantiate DataSources ---
        data_sources = {}
        if "feats" in input_types:
            data_sources["feats"] = FeatDataSource(feat_dir)

        if "labels" in input_types:
            # We use the SAME label map for input features
            data_sources["labels"] = LabelDataSource(seg_to_labels, attribute_index)

        datasets[subset] = SeqEnhDataset(
            subset=subset,
            data_sources=data_sources,
            segment_id_list=valid_ids,
            seq_index=seq_index,
            road_to_seq=road_to_seq,
            context_sequence=context_sequence,
            target_label_map=seg_to_labels,
            target_attribute_index=attribute_index,
        )

    return datasets
