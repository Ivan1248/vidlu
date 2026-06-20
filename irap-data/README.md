# irap_data

Standalone Python package for loading the iRAP road-attribute datasets (iRAP-BiH, iRAP-Vietnam) and an interactive Streamlit-based viewer. This code is based on code by Marin Kačan: see [docs/differences-to-the-original-dataset-code.md](docs/differences-to-the-original-dataset-code.md) for a comparison.


## Installation

From the repository checkout:

```bash
uv pip install -e ./irap_data            # core (loaders only)
uv pip install -e ./irap_data[viewer]    # adds streamlit for the dataset viewer
```

Runtime dependencies: `opencv-python` (image loading and viewer resize) and `torchvision` (`irap_data.jitter` color-jitter augmentation, also used by the viewer).

## Data layout

Set `IRAP_HOME` (or `DATASETS_PATH`) to a parent directory containing:

- iRAP-BiH: `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (metadata JSONs as a sibling)
- iRAP-Vietnam: `IRAP_Vietnam/` (images and metadata together)

Required metadata files (in the metadata dir):

- `splits.json`
- `segment_id_to_data_paths_rel.json`
- `segment_id_to_road_data.json`
- `attribute_metadata.json`
- `road_id_to_segment_id_sequence.json`

### Context-window filtering

Every loader drops segments whose context window (per `context_offsets`) would step off the end of its sequence in `road_id_to_segment_id_sequence.json`, or whose context frames have no image on disk. This is the only context filter `make_vietnam_data()` applies.

`make_bih_data()` additionally restricts segments to the precomputed BiH subsets in `seg_to_res/{train,val,test}.pickle`. These pickles retain only segments that have at least 10 valid context frames on each side within their road sequence, so the labeled-set size is stable for any `max(abs(context_offsets)) ≤ 10`. Pass `use_ncontext_filter=False` to skip the pre-computed filter. The iRAP-Vietnam dataset does not provide such a filter.

### Unlabeled splits

`splits.json` may contain extra keys:

- `unlabeled_train`, `unlabeled_val`, `unlabeled_test` – unlabeled segments assigned to a split by geography (unlabeled segments from the same sequences or geographic areas as the labeled subsets).
- `unlabeled_unlocated` – unlabeled segments from folders with unknown geographic relationship to other splits.

When present, `make_vietnam_data` includes these splits in the returned dict. Unlabeled subsets are loaded with `allow_missing_attributes=True`, so every sample's `target` is an all-`-1` tensor compatible with `torch.nn.CrossEntropyLoss(ignore_index=-1)`. The context-window check still applies; the BiH `seg_to_res/*.pickle` allow-list is bypassed because it only enumerates labeled segments.

## Quickstart

```python
from irap_data import make_bih_data, make_vietnam_data

# iRAP-BiH (uses $IRAP_HOME by default)
splits = make_bih_data(context_offsets=(0, -1, -4))
train = splits["train"]
print(len(train), train.info.class_counts)

example = train[0]
# example: {"rgb": (S, 3, H, W) float tensor in [0, 1],
#           "target": (A,) LongTensor, -1 = unlabeled,
#           "segment_id": str, "sequence_id": str}

# iRAP-Vietnam (loose labels – missing attributes become -1)
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
├── image_utils.py             # load_image_cv2, center_crop, resize_to_cover, hwc_to_chw_float_tensor
├── lazy_dict.py               # LazyDict, Lazy
├── vis_utils.py               # AttributeMetadataDecoder + color/composite helpers
├── jitter.py                  # make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
└── dataset_viewer.py          # Streamlit viewer (entry point)
```

## Conventions

- All public constants for normalization (`RGB_MEAN`, `RGB_STD`, `INPUT_DIM`) and the ignore sentinel (`IGNORE_LABEL_INDEX = -1`) live in `irap_data.irap_dataset`.
- `Dataset.info` is a `LazyDict` with attribute access – both `info["class_counts"]` and `info.class_counts` work. Entries wrapped in `Lazy(...)` are computed on first access.
- Targets use `-1` as the ignore index (matches `torch.nn.CrossEntropyLoss(ignore_index=-1)`).
