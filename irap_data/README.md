# irap_data

Standalone Python package for loading the IRAP road-attribute datasets (IRAP-BiH, IRAP-Vietnam) and an interactive Streamlit-based viewer.

## Installation

From the repository checkout:

```bash
pip install -e ./irap_data            # core (loaders only)
pip install -e ./irap_data[viewer]    # adds streamlit for the dataset viewer
```

`opencv-python` is a runtime dependency for image loading and the viewer's context-strip resize. `torchvision` is used by `irap_data.jitter` for `ColorJitter`-based augmentation (also exercised by the viewer's preview).

## Data layout

Set `IRAP_HOME` (or `DATASETS_PATH`) to a parent directory containing:

- IRAP-BiH: `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (metadata JSONs as a sibling)
- IRAP-Vietnam: `IRAP_Vietnam/` (images and metadata together)

Required metadata files (in the metadata dir):

- `splits.json`
- `segment_id_to_data_paths_rel.json`
- `segment_id_to_road_data.json`
- `attribute_metadata.json`
- `road_id_to_segment_id_sequence.json`

### Context-window filtering

Every loader drops segments whose context window (per `context_offsets`) would step off the end of its road in `road_id_to_segment_id_sequence.json`, or whose context frames have no image on disk. This is the only context filter `make_vietnam_data()` applies.

`make_bih_data()` additionally restricts segments to the precomputed BiH subsets in `seg_to_res/{train,val,test}.pickle`. These pickles are built for a fixed maximum context window of `N = 10` (i.e. only segments with 10 valid neighbors on each side are kept). This means that the labeled-set size stays constant as lon as `max(abs(context_offsets)) ≤ 10`. Pass `use_ncontext_filter=False` to skip the pre-computed filter from the `.pickle` files. The iRAP-Vietnam dataset doesn't provide such apre-computed filter.

### Unlabeled splits

`splits.json` may contain extra keys:

- `unlabeled_train`, `unlabeled_val`, `unlabeled_test` – unlabeled segments assigned to a split by geography (unlabeled segments from the same sequences or geographic areas as the labeled subsets).
- `unlabeled_unlocated` – unlabeled segments from folders with unknown geographic relationship to other splits.

When these keys are present, `make_vietnam_data` returns them alongside the labeled splits in the same dict. Unlabeled subsets are loaded with `allow_missing_attributes=True` automatically, so every sample's `target` is an all-`IGNORE_LABEL_INDEX (-1)` tensor compatible with `torch.nn.CrossEntropyLoss(ignore_index=-1)`. The per-segment context-window check (see *Context-window filtering* above) still applies. The BiH `seg_to_res/*.pickle` allow-list is bypassed for unlabeled subsets because it only enumerates labeled segments.

## Quickstart

```python
from irap_data import make_bih_data, make_vietnam_data

# IRAP-BiH (uses $IRAP_HOME by default)
splits = make_bih_data(context_offsets=(0, -1, -4))
train = splits["train"]
print(len(train), train.info.class_counts)

example = train[0]
# example: {"rgb": (S, 3, H, W) float tensor in [0, 1],
#           "target": (A,) LongTensor, -1 = unlabeled,
#           "segment_id": str, "sequence_id": str}

# IRAP-Vietnam (loose labels – missing attributes become -1)
vn = make_vietnam_data(dataset_dir="/data/IRAP_Vietnam")
```

Inference on a folder of images (no labels required):

```python
from irap_data import InferenceImageDataset

ds = InferenceImageDataset.from_folder(
    "/path/to/images",
    reference_dataset=splits["test"],   # borrows attribute metadata
    context_offsets=(0, -1, -4),
)
```

## Dataset viewer

```bash
IRAP_HOME=/path/to/IRAP_HOME streamlit run irap_data/irap_data/dataset_viewer.py
```

Or, once `irap_data` is installed:

```bash
python -m streamlit run -m irap_data.dataset_viewer
```

The viewer auto-detects which datasets exist under `$IRAP_HOME`, exposes per-attribute filtering, navigates by index or random sample, and previews the surrounding road sequence for each segment.

## Package contents

```
irap_data/
├── __init__.py                # Public API
├── dataset.py                 # Dataset base class + transformations (map/filter/zip/...)
├── irap_dataset.py            # IRAPDataset, make_bih_data, make_vietnam_data, IGNORE_LABEL_INDEX
├── inference_dataset.py       # InferenceImageDataset (label-free folder loader)
├── attrs.py                   # ATTRS_TO_INCLUDE + name-to-index helpers
├── attribute_frequencies.py   # AttributeFrequencyStats, compute_attribute_frequency_stats
├── image_utils.py             # load_image_cv2, rgb_to_chw_tensor
├── lazy_dict.py               # LazyDict, Lazy
├── vis_utils.py               # AttributeMetadataDecoder + color/composite helpers
├── jitter.py                  # make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
└── dataset_viewer.py          # Streamlit viewer (entry point)
```

## Conventions

- All public constants for normalization (`RGB_MEAN`, `RGB_STD`, `INPUT_DIM`) and the ignore sentinel (`IGNORE_LABEL_INDEX = -1`) live in `irap_data.irap_dataset`.
- `Dataset.info` is a `LazyDict` with attribute access – both `info["class_counts"]` and `info.class_counts` work. Entries wrapped in `Lazy(...)` are computed on first access.
- Targets use `-1` as the ignore index (matches `torch.nn.CrossEntropyLoss(ignore_index=-1)`).
