# vidlu_irap_gaim

ViDLU extension for the IRAP GAIM local attribute recognition workflow on road segments (BiH dataset).

The extension is discovered by ViDLU via the `vidlu_` extension naming convention:
- Python package name: `vidlu_irap_gaim`
- Factory namespace name: `irap_gaim`

## Install / import

- If you run from the repository checkout, make sure the repo root is on `PYTHONPATH` so `vidlu_irap_gaim` is importable.
- No separate install step is required.

## Data layout

Set `IRAP_HOME` to a directory that contains:
- `IRAP_BIH/` (images)
- `IRAP_BIH_METADATA/` (metadata JSONs)

The metadata directory is expected to include (at minimum):
- `splits.json`
- `segment_id_to_data_paths_rel.json`
- `segment_id_to_road_data.json`
- `attribute_metadata.json`
- `road_id_to_segment_id_sequence.json`

### N-context filtering (default behavior)

`irap_gaim.make_bih_data()` applies an N-context filter by default (`use_ncontext_filter=True`) using precomputed pickle files:
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/train.pickle`
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/val.pickle`
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/test.pickle`

To disable this filtering (use all segments that pass label/context checks):

```bash
python scripts/run.py train \
  "irap_gaim.make_bih_data(use_ncontext_filter=False)" \
  "standardize" \
  "..." "..."  # (model/trainer as below)
```

## Quickstart (training)

**Important**: the dataset returns RGB in \([0,1]\) and the default pipeline expects normalization via the **`standardize` input adapter** (it uses `dataset.info.pixel_stats`).

### ResNet encoder (Vistas-pretrained backbone)

This matches the current code path: deterministic loading + center crop in the dataset, photometric jitter in the trainer (`irap_gaim.irap_local_rec_trainer`).

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -e vistas_rn18_s3
```

**Where to put `vistas.pt`**: place it at `<VIDLU_PRETRAINED>/irap_gaim/vistas.pt` so `--params "...:irap_gaim/vistas.pt"` resolves correctly.

### ResNet encoder (ImageNet-pretrained backbone only)

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=True)" \
  "irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -e imagenet_rn18_s3
```

### Metrics and attribute filtering

- `irap_gaim.get_irap_metrics(...)` configures metrics over the canonical paper subset (`attrs_to_include`) by mapping attribute names to indices using `dataset.info.attribute_names`.
- Metrics used here **require** `attrs_idx` internally; that’s why using `get_irap_metrics()` is recommended.

## Semi-Supervised Learning with Pseudo-Labels

This extension supports FixMatch-style pseudo-label self-training for leveraging unlabeled data. A frozen pre-trained teacher generates hard argmax pseudo-labels with per-attribute confidence thresholding and temperature scaling.

### Quick start: On-the-fly pseudo-labeling (teacher runs each batch)

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=0.1)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_pseudo_label_trainer,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='/path/to/pretrained_checkpoint.pth',conf_thresh=0.8,temperature=1.0)" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -e 1
```

The trainer specification inherits from the base `irap_pseudo_label_trainer` config and overrides just the `train_step` with custom parameters. You can also use adaptive per-attribute thresholding:

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=0.1)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_pseudo_label_trainer,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='/path/to/pretrained_checkpoint.pth',conf_thresh={0: 0.0, 1: 0.0, 2: 0.0},temperature=1.0)" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -e 1
```

### Offline pseudo-label generation (one-time preprocessing)

First, generate pseudo-labels for the unlabeled set using a pre-trained teacher:

```python
# Example script: generate_pseudo_labels.py
import torch
from vidlu_irap_gaim.tools.generate_pseudo_labels import generate_pseudo_labels, save_pseudo_labels
from vidlu_irap_gaim import make_bih_data

# Load dataset and model
data = make_bih_data()
dataset_unlabeled = data['train_u']
model = torch.load('pretrained_model.pth')  # your pre-trained checkpoint

# Generate pseudo-labels with fixed thresholding
result = generate_pseudo_labels(
    model,
    dataset_unlabeled,
    conf_thresh=0.8,       # mask predictions below 80% confidence
    temperature=1.0,        # standard softmax (no temperature scaling)
    batch_size=32,
    device='cuda'
)
save_pseudo_labels(result, 'pseudo_labels_fixed.npz')
```

Then train on labeled + pseudo-labeled data:

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=0.1)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_pseudo_label_offline_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -e 1
```

### Confidence thresholding heuristics

#### Fixed global threshold

All attributes use the same confidence threshold. Best for balanced datasets.

```python
conf_thresh = 0.8  # (float) all attributes masked below 80% confidence
```

#### Per-attribute adaptive thresholding (MC-PanDA++ style)

Each attribute gets its own threshold updated via EMA based on observed confidence distribution. Recommended for handling class imbalance within attributes.

Initialize with a dict mapping attribute indices to thresholds (starting at 0.0). They adapt per iteration:

```python
# Initialize with zeros; thresholds adapt per batch during training
conf_thresh = {i: 0.0 for i in range(41)}  # dict with one threshold per attribute
```

Pass it in the trainer specification:

```bash
"irap_gaim.irap_pseudo_label_trainer,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='...',conf_thresh={0: 0.0, 1: 0.0, ..., 40: 0.0},temperature=1.0)"
```

For offline mode, pre-compute with a fixed threshold:

```python
result = generate_pseudo_labels(model, dataset, conf_thresh=0.5, ...)
```

### Temperature scaling

Sharpen or soften confidence estimates before thresholding. Especially useful with adaptive thresholds.

```python
# Standard (no scaling)
temperature = 1.0

# Sharpen confidence (more selective, fewer pseudo-labels)
temperature = 0.8

# Soften confidence (less selective, more pseudo-labels)
temperature = 1.2
```

### More commands

- **Dataset viewer**: `IRAP_HOME=/path/to/IRAP_HOME streamlit run vidlu_irap_gaim/tools/dataset_viewer.py`
- See `commands.md` for additional recipes (ViT/DINOv2 encoder, legacy inference).

## Inference and Visualization

We use `vidlu_irap_gaim.tools.inference` to run model evaluation and generate rich visualizations (images with predicted attributes, colored probability bars, and ground truth comparison).

### Evaluation on standard splits

To run inference on the `test` split and save visualizations:

```bash
VIDLU_DETAILED_EVAL=1 IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py test \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" \
  -r best \
  -m "irap_gaim.tools.inference"
```

- This will create a `visualizations/test` directory inside your experiment folder.
- Results include a `predictions.json` and PNG images for each sample.

### Inference on a custom image folder (unlabeled)

You can run the model on any folder of images (sorted alphabetically) using `InferenceImageDataset`:

```bash
VIDLU_DETAILED_EVAL=1 IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py test \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,..." \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim.tools.inference:run,e,dataset=irap_gaim.InferenceImageDataset.from_folder('/path/to/images',reference_dataset=e.data.test,context_sequence=(0,-1,-4))"
```

- `InferenceImageDataset.from_folder` automatically detects unlabeled images and skips loss/metrics computation.
- `reference_dataset=e.data.test` is used to copy attribute metadata and pixel normalization stats.

## Feature export (optional)

Export per-segment features (for sequential enhancement / smoothing):

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py test \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim:export_feats,split='val',feat_dir='FEATS/val'"
```

**Requirement**: the model must support `forward(..., return_features=True)` (the provided `ImageSequenceClassifier` does).

## Sequential enhancement (optional)

`irap_gaim.make_seq_enh_data(...)` builds a per-attribute sequence dataset from exported `.npy` features.

`irap_gaim.GeneralLSTMModel` expects **`input_encoders`**, not a raw `input_dim`. Example (features-only input):

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_seq_enh_data(feat_dir='FEATS/train',attribute=0)" \
  "id" \
  "irap_gaim.GeneralLSTMModel,n_classes=<N_CLASSES>,input_encoders=dict(feats=irap_gaim.IdentityEncoder(input_dim=<FEATURE_DIM>))" \
  "ct.classification" \
  --metrics "A"
```

Notes:
- Replace `<FEATURE_DIM>` with the flattened dimension of one exported `*.npy`.
- Replace `<N_CLASSES>` with the class count for the chosen attribute.

## Key components (API sketch)

- **Data**:
  - `irap_gaim.make_bih_data(..., use_ncontext_filter=True, seg_to_res_path=None)`
  - `irap_gaim.BihSequence(...)` (populates `info.class_counts`, `info.pixel_stats`, `info.attribute_names`)
  - `irap_gaim.get_class_counts(...)`
- **Models**:
  - `irap_gaim.ImageSequenceClassifier(class_counts, sequence_length, attention=False, encoder_f=...)`
  - `irap_gaim.ResNetEncoder(pretrained=True|False, ...)`
  - `irap_gaim.dinov2_vit_encoder(...)`
- **Training**:
  - `irap_gaim.irap_local_rec_trainer` (freeze-then-finetune schedule + color jitter + dynamic balanced recall weights)
- **Metrics**:
  - `irap_gaim.get_irap_metrics(dataset=None, class_counts=None, attrs_to_include=None)`
- **Pretraining helper**:
  - `irap_gaim.vistas_params_spec(...)` (Python helper to construct a `--params` translation string)

## Troubleshooting (high-signal)

- **Missing `seg_to_res/*.pickle`**:
  - Either create the pickle files under `IRAP_BIH_METADATA/seg_to_res/`, or disable filtering with `make_bih_data(use_ncontext_filter=False)`.
- **`vistas.pt` not found**:
  - Put it at `<VIDLU_PRETRAINED>/irap_gaim/vistas.pt`, or use an absolute path in the `--params` string.
