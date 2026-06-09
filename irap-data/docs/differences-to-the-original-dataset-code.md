# Equivalence with `irap_gaim/dataset_wrapper.py`

Date: 2026-05-24

This document compares `irap_data/irap_data/irap_dataset.py` (`IRAPDataset` / `make_bih_data`) against the original [`irap_gaim/dataset_wrapper.py`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py) (`DatasetWrapper`), [`image_sequence_dataset.py`](https://github.com/mkacan/irap_gaim/blob/main/image_sequence_dataset.py) (`ImageSequenceDataset`), and the [`train_local_rec.py`](https://github.com/mkacan/irap_gaim/blob/main/train_local_rec.py) driver. The comparison is restricted to the **standard supervised iRAP-BiH configuration** defined by the original [`irap_gaim/config.json`](https://github.com/mkacan/irap_gaim/blob/main/config.json):

```jsonc
// excerpt
"context_sequence":          [0, -1, -4],
"data_types":                ["rgb"],
"input_dim": { "rgb":        [384, 288, 3] },
"attribute_value_mapping_path": "",        // -> identity mapping branch
"normalization_statistics":  { "mean": [...BiH...], "std": [...BiH...] }
```

`make_bih_data()` with all defaults is the new-code counterpart of that config (`context_offsets = (0, -1, -4)`, `use_ncontext_filter = True`, `allow_missing_attributes = False`, `mean/std = RGB_MEAN/STD`, `input_dim_rgb = (384, 288, 3)`, `transforms = None`).

## Equivalent behavior

| Step | Original | New |
|---|---|---|
| Drop split entries with no data path | [`dataset_wrapper.py:13-22`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L13-L22) | `irap_dataset.py:388-402` |
| `seg_to_res/{split}.pickle` N-context filter, `max_N = 10`, per-split, union over splits | [`train_local_rec.py:196-229`](https://github.com/mkacan/irap_gaim/blob/main/train_local_rec.py#L196-L229) → [`n_context_dataset.py:4-35`](https://github.com/mkacan/irap_gaim/blob/main/n_context_dataset.py#L4-L35) | `irap_dataset.py:60-156` |
| Label construction (identity remap) | [`dataset_wrapper.py:189-272`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L189-L272) (`_load_default_attribute_metadata` + `_map_attribute_values` + `_get_segment_id_to_attribute_value`) | `irap_dataset.py:412-454` |
| Per-attribute class count | `len(attribute_value_to_irap_number[attr])` ([`dataset_wrapper.py:289-294`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L289-L294)) | same expression |
| Attribute ordering | sorted by `attribute_to_idx` ([`dataset_wrapper.py:355`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L355)) | `idx_to_attribute[i] for i in range(...)` |
| Drop segments with unknown iRAP codes | `KeyError` in [`dataset_wrapper.py:256-272`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L256-L272) | `attr_irap_to_value[attr].get(irap_code) is None` branch |
| Filter splits by valid context window | [`dataset_wrapper.py:38-55`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L38-L55) | `irap_dataset.py:477-490` |
| Apply `ncontext_segment_id_subset` | [`dataset_wrapper.py:57-62`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L57-L62) | `irap_dataset.py:494-501` |
| Context-frame offsets `(0, -1, -4)` ordering | iterated in `context_sequence` order (`config.json` key) | iterated in `context_offsets` order |

[`_remove_filtered_out_segments`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L370-L381) is a no-op under identity remap, so the new version's omission of it is equivalent.

The following differences also have no effect under the standard iRAP-BiH configuration:

- **Filtering order.** The original ([`dataset_wrapper.py:13-91`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L13-L91)) filters in the order data paths → context → `ncontext_subset` → label mapping → two more `_filter_segments` passes. The new code filters in the order data paths → label mapping → context → `ncontext_subset`. The final intersection is the same.
- **Missing-attribute handling.** The original raises a `KeyError` in [`_create_segment_id_to_labels`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L339-L362) when `required_attributes` is incomplete. The new code drops the segment instead (or labels it `-1` when `allow_missing_attributes=True`). iRAP-BiH has complete attribute coverage, so neither branch fires.
- **Sample-dict shape.** The original ([`image_sequence_dataset.py:52-72`](https://github.com/mkacan/irap_gaim/blob/main/image_sequence_dataset.py#L52-L72)) emits `{rgb, target, segment_id}` plus optional `depth`. The new code emits `{rgb, target, segment_id, sequence_id}` and never `depth`. With `data_types: ["rgb"]` the original also never produces `depth`, and `sequence_id` (road id) is purely additive.

## Behavioral differences

### 1. Context-id resolution mechanism

| | Original | New |
|---|---|---|
| Method | integer arithmetic on segment IDs: `str(int(sid) + offset)` | road-sequence lookup via `road_id_to_segment_id_sequence.json` |
| Validity domain for a context frame | `int(seg_id) ∈ set(map(int, splits['all']))` (i.e. labelled `train ∪ val ∪ test` after the data-path filter) | `cid ∈ seg_to_paths` (every segment with an image on disk) |

The two coincide on iRAP-BiH iff (a) segment IDs are integer-contiguous within each road, (b) IDs across roads are not numerically adjacent, and (c) every `segment_id_to_data_paths_rel.json` entry appears in some labelled split. The road-based lookup cannot pull a frame from a different road, which allows the same code to handle iRAP-Vietnam. Under (a)–(c) the two are expected to match end-to-end, but this has not been verified.

### 2. Image resize / crop

| | Original ([`dataset_wrapper.py:141-144`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L141-L144)) | New (`image_utils.py:rgb_to_chw_tensor`) |
|---|---|---|
| Resize | none | resize-to-cover (Lanczos, preserves aspect ratio) when source < target |
| Crop | `CenterCrop((H, W))` only when `input_dim['rgb'] != [384, 284, 3]` | always center-crop to `target_wh` |

The `[384, 284, 3]` literal looks like a typo for `288`: `input_dim.rgb` is `[384, 288, 3]`, so the `!=` branch fires and the original does crop.

The new code's resize is guarded by `if (new_w, new_h) != (w, h):` and fires only when some source dimension is strictly below the corresponding target. Sources at or above `target_wh` in both dimensions take the crop-only path, matching the original. At exactly `384 × 288` the crop is full-bounds, so pixels pass through unchanged (the only remaining difference is that the original emits a standardized tensor rather than a `[0, 1]` `float32` — see §3).

The two paths therefore diverge only on under-target sources. Resize-to-cover is retained for iRAP-Vietnam, whose frame sizes have not been confirmed.

### 3. Augmentation and normalization moved out of the dataset

The original applies, inside `__getitem__`:

- [`ColorJitter(0.6, 0.3, 0.2, 0.02)`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L139-L140) on the `train` split, and
- [`Normalize(mean, std)`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L145) on every split, with iRAP-BiH mean/std from [`config.json:11-15`](https://github.com/mkacan/irap_gaim/blob/main/config.json#L11-L15).

`IRAPDataset` does neither: `rgb` is deterministic, unnormalized, `float32` in `[0, 1]`. The iRAP-BiH mean/std are exposed as `RGB_MEAN` / `RGB_STD` and via `info.pixel_stats`, so the caller owns augmentation and standardization. To match the original numerically, the caller must apply that same `ColorJitter` to the "train" split and standardize every split with `info.pixel_stats` — outside `irap_data`'s scope.

## Bottom line

Under the standard iRAP-BiH config, `IRAPDataset` matches the original in **segment selection, per-attribute labels, class counts, and 3-frame ordering**.

Bit-identical sample tensors additionally require:

1. iRAP-BiH segment IDs are integer-contiguous within each road and non-adjacent across roads, so integer arithmetic and road-sequence indexing agree.
2. Source images are already at `384 × 288`, so the resize is a no-op.
3. The caller applies `ColorJitter(0.6, 0.3, 0.2, 0.02)` to the train split and standardizes with `info.pixel_stats`.

(1) and (2) are properties of the on-disk release. (3) is outside `irap_data`'s scope.
