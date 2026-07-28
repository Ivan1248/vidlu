# vidlu_irap_gaim

ViDLU extension for iRAP attribute recognition on road segments. The code is based on https://github.com/mkacan/irap_gaim [1].

The extension is discovered by ViDLU via the `vidlu_` extension naming convention:
- Python package name: `vidlu_irap_gaim`
- Factory namespace name: `irap_gaim`

## Table of contents

- [Installation & requirements](#installation--requirements)
- [Data layout](#data-layout)
- [Package structure](#package-structure)
- [Quickstart (supervised training)](#quickstart-supervised-training)
- [Encoders](#encoders)
- [Metrics & dynamic weighting](#metrics--dynamic-weighting)
- [Semi-supervised learning with pseudo-labels](#semi-supervised-learning-with-pseudo-labels)
- [Joint BiH + Vietnam training](#joint-bih--vietnam-training)
- [Multi-scale inference](#multi-scale-inference)
- [VLM integration (zero-shot & fine-tuning)](#vlm-integration-zero-shot--fine-tuning)
- [Inference & visualization](#inference--visualization)
- [Feature export & sequential enhancement](#feature-export--sequential-enhancement)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Installation & requirements

If you run from the repository checkout, make sure the repo root is on `PYTHONPATH` so `vidlu_irap_gaim` is importable. No separate install step is required.

Core dependencies (see `requirements.txt`):

```
numpy, opencv-python, pillow, tqdm
torch>=1.9.0, torchvision>=0.9.0
matplotlib, scikit-learn, streamlit
```

For VLM inference (optional):

```
transformers>=4.40.0, qwen-vl-utils>=0.0.8, pyyaml>=6.0
```

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
  "..." "..."
```

## Package structure

```
vidlu_irap_gaim/
├── __init__.py                  # Public API exports
├── losses.py                    # MultiAttributeCrossEntropyLoss
├── metrics.py                   # Per-attribute accuracy, precision, recall, F1, IoU
├── data/                        # Dataset and attribute management
│   ├── irap_dataset.py           # IRAPDataset dataset, make_bih_data factory
│   ├── inference_dataset.py     # InferenceImageDataset for unlabeled data
│   ├── attrs.py                 # Canonical 41-attribute subset definitions
│   ├── attribute_frequencies.py # Class distribution analysis
│   └── constants.py             # RGB normalization constants
├── models/                      # Neural network models and encoders
│   ├── classification.py        # ImageSequenceClassifier
│   ├── multiscale.py            # MultiScaleSequenceInference
│   ├── pretraining.py           # Vistas pre-training helpers
│   ├── resnet_backbone.py       # Legacy ResNet implementation
│   └── encoders/
│       ├── resnet.py            # ResNetEncoder (ImageNet / Vistas)
│       ├── vit.py               # ViTEncoder, dinov2_vit_encoder
│       └── attention.py         # Attention pooling
├── training/                    # Training infrastructure
│   ├── configs.py               # Trainer configurations (supervised, semi-sup, pseudo-label, VLM)
│   ├── steps.py                 # MultiScaleSupervisedStep, MultiAttributePseudoLabelStep
│   ├── extensions.py            # FreezeThenFinetune, MultiAttributeScorePrinter, VisualizationExtension
│   ├── dynamic_weights.py       # DynamicBalancedRecallWeights (per-epoch class reweighting)
│   ├── semisup.py               # Semi-supervised splits, pseudo-label generation, adaptive thresholds
│   ├── jitter.py                # Color jitter augmentation
│   └── helpers.py               # Trainer configuration helpers
├── seq/                         # Sequential enhancement (LSTM smoothing)
│   ├── dataset.py               # SeqEnhDataset, make_seq_enh_data factory
│   ├── models.py                # GeneralLSTMModel for temporal smoothing
│   └── feats.py                 # Feature export to .npy files
├── vlm/                         # Vision-Language Model integration
│   ├── base.py                  # BaseVLMPredictor, VLMPredictionResult
│   ├── qwen3_vl.py              # Qwen3VLPredictor (HuggingFace)
│   ├── qwen3_vl_vllm.py         # Qwen3VLvLLMPredictor (vLLM engine)
│   ├── prompts.py               # PromptBuilder with configurable detail levels
│   ├── response_scheme.py       # Response format schemes (Standard, JSON, Indexed, Sparse)
│   ├── response_parser.py       # Parse VLM text responses to structured predictions
│   ├── attribute_prompts.yaml   # YAML prompt configuration
│   └── finetuning/              # LoRA fine-tuning pipeline
│       ├── model.py             # Qwen3VLClassifier (LoRA wrapper)
│       ├── dataset.py           # VLMIrapDataset, make_vlm_bih_data
│       ├── predictor.py         # FineTunedVLMPredictor
│       └── steps.py             # VLMTrainStep, VLMEvalStep
├── tools/                       # Utility scripts and visualization
│   ├── vis_utils.py             # Visualization utilities (color palettes, composite images)
│   ├── dataset_viewer.py        # Streamlit interactive data browser
│   ├── inference.py             # Evaluation hook for structured predictions
│   ├── inference_visualization.py  # Standalone PNG generation
│   ├── generate_pseudo_labels.py   # Offline pseudo-label generation
│   ├── baseline_random.py       # Random baseline predictor
│   ├── attribute_most_common_report.py  # Attribute frequency analysis
│   ├── vlm_benchmark.py         # VLM benchmarking
│   └── vlm_inference.py         # VLM inference pipeline
├── compat/
│   └── legacy_seq_enh_model.py  # Backward-compatible legacy LSTM models
└── tests/
    ├── test_semisup.py
    ├── test_qwen3.py
    └── test_vlm.py
```

## Quickstart (supervised training)

**Important**: the dataset returns RGB in \([0,1]\) and the default pipeline expects normalization via the **`standardize` input adapter** (it uses `dataset.info.pixel_stats`).

### ResNet encoder (Vistas-pretrained backbone)

Deterministic loading + center crop in the dataset, photometric jitter in the trainer.

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)" \
  -e 1
```

**Where to put `vistas.pt`**: place it at `<VIDLU_PRETRAINED>/irap_gaim/vistas.pt` so `--params "...:irap_gaim/vistas.pt"` resolves correctly.

**Epoch schedule (2 + 8 vs the original 2 + 13)**: `irap_local_rec_trainer` defaults to `epoch_count=10`
(2 frozen + 8 finetune), matching the epoch counts of the original repo's
`train_local_rec_paper_ep10.sh` variant (but not its per-epoch LR decay; see below). The
original *paper* recipe (`train_local_rec.py` argparse defaults and `train_local_rec_paper.sh`) is
2 frozen + **13** finetune = **15** epochs. To reproduce the 15-epoch recipe without any code change,
override the trainer's `epoch_count` (and `eval_count`, so the per-epoch validation recall that
`DynamicBalancedRecallWeights` consumes is still computed every epoch) in the trainer factory string:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer,epoch_count=15,eval_count=15" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

`FreezeThenFinetune` unfreezes at epoch 2 regardless of the total, so `epoch_count=15` yields the
2-frozen + 13-finetune schedule. (Note: `scripts/run.py -e` is the experiment-name suffix, *not* an
epoch count; the `-e 1` in the smoke-test command above only labels the run.)

**LR decay adapts to the epoch count**: `FreezeThenFinetune` fixes each phase's *total* LR decay at
the paper's values (0.8² frozen, 0.88¹³ finetune) and derives the per-epoch multiplicative factor as
`total ** (1/phase_length)`. With `epoch_count=15` this is exactly the paper's 0.8/0.88 per epoch;
the default `epoch_count=10` compresses the same total decay into 8 finetune epochs (≈0.812/epoch),
so it ends at the paper's final LR rather than matching `train_local_rec_paper_ep10.sh`, which keeps
0.88/epoch and therefore ends the shorter run at a higher LR.

### DINOv2 ViT encoder

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.dinov2_vit_encoder,variant='dinov2_vitb14',params_dir=dirs.pretrained)" \
  "irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

### MAE ViT encoder (ImageNet-pretrained)

Loads a Hugging Face `ViTMAEModel` (`facebook/vit-mae-base` by default) as a plain
ViT feature extractor — MAE random masking is disabled (`mask_ratio=0`). It expects
**ImageNet** normalization (the checkpoint's own preprocessor stats), so use
`standardize(imagenet)` rather than the dataset-stats `standardize`:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize(imagenet)" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.mae_vit_encoder,variant='facebook/vit-mae-base',params_dir=dirs.pretrained)" \
  "irap_gaim.irap_local_rec_trainer_nofreeze" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

Use `variant='facebook/vit-mae-large'` / `'-huge'` (or any HF ViT-MAE repo id) for
larger models. `irap_local_rec_trainer` (freeze-then-finetune) also works, since the
encoder's `pooling_parameters()` is empty (frozen phase trains heads only).

### iRAP-Vietnam

iRAP-Vietnam shares the same 41-attribute schema (same `attribute_metadata.json`)
as iRAP-BiH, but does not annotate 7 of the attributes (the flow attributes,
Upgrade cost, Roadworks, Bicycle facility); those columns are the ignore label
`-1` in every Vietnam target, so the loss skips them per example and the metrics
report `n=0` for them.

```bash
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_vietnam_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

Pass the loaded training split to `get_irap_metrics(data.train)`; the no-argument
form silently reloads `make_bih_data()` to resolve the attribute set.

## Encoders

| Encoder | Factory | Pretrained weights | Notes |
|---------|---------|-------------------|-------|
| ResNet-18 | `ResNetEncoder(pretrained=True)` | ImageNet (torchvision) | Good baseline |
| ResNet-18 + Vistas | `ResNetEncoder(pretrained=False)` + `--params` | Vistas `.pt` file | Best for road scenes; load via `--params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt"` |
| DINOv2 ViT-B/14 | `dinov2_vit_encoder(variant='dinov2_vitb14')` | Auto-downloaded | Self-supervised; no `--params` needed |
| MAE ViT-B/16 | `mae_vit_encoder(variant='facebook/vit-mae-base')` | Auto-downloaded (HF) | Masking disabled; use `standardize(imagenet)` |

## Metrics & dynamic weighting

### Metrics

`irap_gaim.get_irap_metrics(...)` configures per-attribute accuracy, precision, recall, F1, and IoU over the canonical 41-attribute subset. It maps attribute names to indices using `dataset.info.attribute_names`.

```bash
--metrics "irap_gaim.get_irap_metrics(data.train)"
```

### Dynamic balanced recall weights

`DynamicBalancedRecallWeights` is a trainer extension that recomputes per-attribute class weights after each validation epoch. The weighting formula balances inverse class frequency with observed recall:

```
w = inv_freq * (1 - recall) + sqrt(inv_freq) * recall
```

This is configured automatically in the standard trainers. It requires `MultiAttributeClassificationMetrics` (which implements `InternalMetricsProvider`) to access per-class TP/FP/FN statistics.

## Semi-supervised learning with pseudo-labels

This extension supports FixMatch-style pseudo-label self-training for leveraging unlabeled data. A frozen pre-trained teacher generates hard argmax pseudo-labels with per-attribute confidence thresholding and temperature scaling.

### Source of the unlabeled set

`make_semisup_data` chooses the unlabeled pool in this order (default `prefer_real_unlabeled=True`):

1. **Real `unlabeled_train` split** from `splits.json` when present (e.g. iRAP-Vietnam after running the prep pipeline). The full labeled `train` split is kept; `labeled_ratio` is ignored.
2. **Synthetic split** of the labeled `train` set by `labeled_ratio` / `labeled_size` – the historical iRAP-BiH behaviour, still used when no `unlabeled_train` key exists.

Pass `prefer_real_unlabeled=False` to force the synthetic path even on metadata that has a real unlabeled split.

### On-the-fly pseudo-labeling (teacher runs each batch)

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_semisup_data(irap_gaim.make_bih_data(), labeled_ratio=0.1)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_pseudo_label_trainer,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='/path/to/checkpoint.pth',conf_thresh=0.8,temperature=1.0)" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)" \
  -e 1
```

### Offline pseudo-label generation (one-time preprocessing)

Generate pseudo-labels, then train on the combined labeled + pseudo-labeled data:

```python
from vidlu_irap_gaim.tools.generate_pseudo_labels import generate_pseudo_labels, save_pseudo_labels

result = generate_pseudo_labels(model, dataset_unlabeled, conf_thresh=0.8, temperature=1.0, batch_size=32, device='cuda')
save_pseudo_labels(result, 'pseudo_labels_fixed.npz')
```

```bash
# Train on labeled + offline pseudo-labels
python scripts/run.py train \
  "irap_gaim.make_semisup_data(irap_gaim.make_bih_data(), labeled_ratio=0.1)" "standardize" \
  "irap_gaim.ImageSequenceClassifier,..." \
  "irap_gaim.irap_pseudo_label_offline_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)" -e 1
```

### Confidence thresholding

| Strategy | `conf_thresh` value | Description |
|----------|-------------------|-------------|
| Fixed global | `0.8` (float) | All attributes use the same threshold. Best for balanced datasets. |
| Per-attribute adaptive | `{0: 0.0, 1: 0.0, ..., 40: 0.0}` (dict) | MC-PanDA++ style: each attribute's threshold adapts via EMA based on observed confidence distribution. Recommended for class imbalance. |

### Temperature scaling

| `temperature` | Effect |
|---------------|--------|
| `1.0` | Standard softmax (no scaling) |
| `< 1.0` (e.g. `0.8`) | Sharpened confidence – more selective, fewer pseudo-labels |
| `> 1.0` (e.g. `1.2`) | Softened confidence – less selective, more pseudo-labels |

## Joint BiH + Vietnam training

Because both releases share the same `attribute_metadata.json`, their per-attribute
class indexing and `class_counts` are identical, and each emits length-41 targets
with unlabeled attributes set to `-1`. So the two datasets are directly
concatenable onto one 41-head model: the loss uses whichever attributes each
example labels (Vietnam examples supervise the 34 shared attributes; BiH examples
supervise all 41). No changes to the model or loss are needed — only how the data
string composes the splits.

The data string is evaluated as Python (a mapping-returning expression, or
multiple statements assigning `data`). Bind each release once and reuse its splits.
`Dataset.join(other, info=...)` concatenates and lets you choose which `info` the
joined split carries. **Prefer Vietnam's `info`** for the joined training split: the
34 shared attributes are what matter in these combined-dataset experiments, and using
Vietnam `info` keeps its `class_counts`-derived weighting focused on them. The model
still has 41 heads either way, since `class_counts` is the full schema in both infos.

Evaluation metrics are chosen **per split** (see
[Per-split evaluation metrics](#per-split-evaluation-metrics)): pass a
`dict(split_name=..., ...)` to `--metrics` to score `val_vn` on the 34 shared
attributes (NaN-free) and `val_bih` on all 41. The first `val*` split in the `data`
dict drives checkpoint selection, so list the Vietnam split first.

### Supervised, concatenated (proportions ∝ split sizes)

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "b = irap_gaim.make_bih_data(); v = irap_gaim.make_vietnam_data(); data = dict(train=b.train.join(v.train, info=v.train.info), val_vn=v.val, val_bih=b.val)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "dict(val_vn=irap_gaim.get_irap_metrics(data.val_vn), val_bih=irap_gaim.get_irap_metrics(data.val_bih))"
```

### Supervised, constant per-batch proportions

Keep the two datasets as separate `train*` splits and use `combined_train_loader_f`,
which draws a fixed count from each per batch (`batch_size=[bih, vietnam]`, matched
to the split order in the dict). An epoch covers the larger dataset once while the
smaller repeats in full shuffled passes, so no example is revisited before the rest
of its dataset's pass.

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "b = irap_gaim.make_bih_data(); v = irap_gaim.make_vietnam_data(); data = dict(train_bih=b.train, train_vn=v.train, val_vn=v.val, val_bih=b.val)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train_vn.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer,data_loader_f=irap_gaim.combined_train_loader_f,batch_size=[8,4]" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "dict(val_vn=irap_gaim.get_irap_metrics(data.val_vn), val_bih=irap_gaim.get_irap_metrics(data.val_bih))"
```

### Semi-supervised (other release as the unlabeled pool)

Put the second release under `train_u` and use a semi-supervised trainer
(`irap_semisup_trainer`, `irap_pseudo_label_trainer`, ...); the consistency /
pseudo-label step consumes `train_u` and ignores its labels.

TODO: all available labels should be used, and additional unlabeled Vietnam splits should be used for the unlabeled data pool. The below command only uses the labeled Vietnam split as the unlabeled pool, which is not good.

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "b = irap_gaim.make_bih_data(); v = irap_gaim.make_vietnam_data(); data = dict(train=b.train, train_u=v.train, val_vn=v.val, val_bih=b.val)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_semisup_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "dict(val_vn=irap_gaim.get_irap_metrics(data.val_vn), val_bih=irap_gaim.get_irap_metrics(data.val_bih))"
```

### Per-split evaluation metrics

`get_irap_metrics(dataset)` scores exactly the attributes that have at least one
labeled example in `dataset` — it computes
`filter_labeled_attrs(get_attrs_to_include(), dataset.info.attr_to_num_labeled)`.
Because both releases share the schema, `class_counts` and the attribute→index map
are identical; the **only** thing the argument changes is this attribute set:

- A **Vietnam** split (`data.val_vn`) → the **34** shared attributes. All 34 are
  labeled in both releases, so the split is NaN-free and its aggregate
  `amF1`/`amP`/`amR` (used for checkpoint selection) stays finite.
- A **BiH** split (`data.val_bih`) → all **41** attributes. On a *Vietnam* split this
  would reintroduce NaN (the 7 BiH-only attributes have no labeled Vietnam example),
  but on `val_bih` itself every attribute is labeled, so all 41 are valid.

To score each split with the attribute set that suits it, pass `--metrics` a
**mapping** from split name to metrics instead of a single list:

```bash
--metrics "dict(val_vn=irap_gaim.get_irap_metrics(data.val_vn), val_bih=irap_gaim.get_irap_metrics(data.val_bih))"
```

Each listed split is evaluated with its own metrics (`val_vn`: 34 attributes,
`val_bih`: 41). A split not named in the mapping — and the training-progress display
— falls back to the **first** entry's metrics (here `val_vn`), so list the split
whose metrics should also apply to training first. The single-list form
(`--metrics "irap_gaim.get_irap_metrics(data.val_vn)"`) still works and applies the
same metrics to every split.

**Checkpoint selection** uses the first `val*` split in the `data` dict (its
`amF1`), unless overridden with `checkpoint_split_prefix`. List the Vietnam split
first (`data = dict(..., val_vn=v.val, val_bih=b.val)`) to checkpoint on Vietnam.

## Multi-scale inference

`MultiScaleSequenceInference` wraps an `ImageSequenceClassifier` and applies it at multiple scales (default: 1.0, 0.75, 1/0.75), averaging probabilities across scales for each attribute.

```python
from vidlu_irap_gaim import MultiScaleSequenceInference

ms_model = MultiScaleSequenceInference(base_model, scales=(1.0, 0.75, 1/0.75))
probs = ms_model(x)  # x: (B, S, C, H, W) -> tuple of (B, K_i) probability tensors
```

For training with multi-scale supervision, use the `irap_local_rec_trainer_multiscale` trainer or the `MultiScaleSupervisedStep` train step.

## VLM integration (zero-shot & fine-tuning)

The `vlm/` subpackage integrates vision-language models (such as Qwen3-VL) for zero-shot and fine-tuned road attribute classification.

### Zero-shot inference

Two predictor backends are available:

| Predictor | Backend | Best for |
|-----------|---------|----------|
| `Qwen3VLPredictor` | HuggingFace Transformers | Single-GPU, small-scale |
| `Qwen3VLvLLMPredictor` | vLLM engine | Batched inference, prefix caching |

```python
from vidlu_irap_gaim.vlm import Qwen3VLPredictor

predictor = Qwen3VLPredictor(model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda")
result = predictor.predict(image, attribute_names, attr_to_value_to_class_idx)
# result.predictions: dict[str, AttributePrediction]
```

### Prompt configuration

`PromptBuilder` supports multiple detail levels for prompt construction:

| Detail level | Content |
|-------------|---------|
| `attr_desc_vals` | Attribute description + valid values + default (most verbose, default) |
| `attr_vals` | Attribute name + valid values + default |
| `attr` | Attribute names only |
| `none` | Empty preamble |

Prompts can be configured via `attribute_prompts.yaml` without code changes.

### Response schemes

Response parsing supports multiple formats via `ResponseScheme` subclasses:

- `StandardResponseScheme` – plain text attribute-value pairs
- `JsonResponseScheme` – structured JSON output
- `IndexedResponseScheme` – numbered attribute-value pairs
- `SparseStandardResponseScheme` / `SparseIndexedResponseScheme` – only non-default values

### Fine-tuning with LoRA

`Qwen3VLClassifier` wraps Qwen3-VL with LoRA adapters for Vidlu training integration:

```bash
python scripts/run.py train "irap_gaim.make_vlm_bih_data(detail_level='attr_desc_vals', response_scheme='standard')" "id" "irap_gaim.Qwen3VLClassifier,model_id='Qwen/Qwen3-VL-8B-Instruct',lora_r=64" "irap_gaim.vlm_finetune_trainer"
```

Key design: adapter-only state dict (~100MB vs ~16GB full model), eager loading for optimizer compatibility, 4-bit quantization support.

### VLM tools

- **`tools/vlm_inference.py`** – full VLM inference pipeline on BiH data
- **`tools/vlm_benchmark.py`** – benchmarking script (prefix caching, throughput)

## Inference & visualization

### Evaluation on standard splits

```bash
VIDLU_DETAILED_EVAL=1 IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py test \
  "irap_gaim.make_bih_data()" "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)" \
  -r best \
  -m "irap_gaim.tools.inference"
```

Creates a `visualizations/test` directory with `predictions.json` and PNG images per sample.

### Inference on a custom image folder (unlabeled)

```bash
python scripts/run.py test \
  "irap_gaim.make_bih_data()" "standardize" \
  "irap_gaim.ImageSequenceClassifier,..." \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim.tools.inference:run,e,dataset=irap_gaim.InferenceImageDataset.from_folder('/path/to/images',reference_dataset=e.data.test,context_offsets=(0,-1,-4))"
```

`InferenceImageDataset.from_folder` detects unlabeled images and skips loss/metrics computation. Use `reference_dataset` to copy attribute metadata and pixel normalization stats.

### Standalone visualization tool

Generate per-segment PNGs with predicted attributes, colored probability bars, and ground truth comparison:

```bash
python vidlu_irap_gaim/tools/inference_visualization.py \
  --mode local \
  --split val \
  --context_offsets "0,-1,-4" \
  --input_adapter standardize \
  --checkpoint_dir "/path/to/checkpoint" \
  --output_dir visualization_output \
  --limit 50 --verbose
```

For legacy sequential enhancement models, use `--mode sequential_legacy` with `--seq_config_path`, `--seq_models_root`, and `--feat_dir`.

### Dataset viewer (Streamlit)

```bash
IRAP_HOME=/path/to/IRAP_HOME streamlit run irap_data/irap_data/dataset_viewer.py
```

## Feature export & sequential enhancement

### Feature export

Export per-segment features (for sequential enhancement / smoothing):

```bash
python scripts/run.py test \
  "irap_gaim.make_bih_data()" "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim:export_feats,split='val',feat_dir='FEATS/val'"
```

The model must support `forward(..., return_features=True)` (the provided `ImageSequenceClassifier` does).

### Sequential enhancement (LSTM smoothing)

`irap_gaim.make_seq_enh_data(...)` builds a per-attribute sequence dataset from exported `.npy` features. `irap_gaim.GeneralLSTMModel` expects `input_encoders`, not a raw `input_dim`:

```bash
python scripts/run.py train \
  "irap_gaim.make_seq_enh_data(feat_dir='FEATS/train',attribute=0)" \
  "id" \
  "irap_gaim.GeneralLSTMModel,n_classes=<N_CLASSES>,input_encoders=dict(feats=irap_gaim.IdentityEncoder(input_dim=<FEATURE_DIM>))" \
  "ct.classification" \
  --metrics "A"
```

Replace `<FEATURE_DIM>` with the flattened dimension of one exported `*.npy`, and `<N_CLASSES>` with the class count for the chosen attribute.

## API reference

### Data

| Symbol | Description |
|--------|-------------|
| `make_irap_data(dataset_dir=, metadata_dir=, ...)` | Generic IRAP loader for any release |
| `make_bih_data(use_ncontext_filter=True, ...)` | Load IRAP-BiH (preset over `make_irap_data`) |
| `make_vietnam_data(...)` | Load IRAP-Vietnam (preset over `make_irap_data`) |
| `make_irap_data_by_name("bih"\|"vietnam", ...)` | Build a release by name (registry dispatch) |
| `make_semisup_data(base_data, labeled_ratio=..., ...)` | Semi-supervised split over a base dataset dict |
| `IRAPDataset(...)` | Dataset class (`info.class_counts`, `info.pixel_stats`, `info.attribute_names`) |
| `InferenceImageDataset.from_folder(...)` | Inference on unlabeled image folders |
| `get_class_counts(...)` | Class count tuple for model construction |
| `make_seq_enh_data(feat_dir, attribute)` | Sequential enhancement dataset from `.npy` features |

### Models

| Symbol | Description |
|--------|-------------|
| `ImageSequenceClassifier(class_counts, sequence_length, attention, encoder_f)` | Temporal sequence classifier |
| `ResNetEncoder(pretrained=True\|False)` | ResNet-18/34/50 backbone |
| `dinov2_vit_encoder(variant, params_dir)` | DINOv2 ViT encoder factory |
| `mae_vit_encoder(variant, params_dir)` | MAE ViT encoder factory (masking disabled; ImageNet norm) |
| `MultiScaleSequenceInference(base_model, scales)` | Multi-scale probability averaging wrapper |
| `GeneralLSTMModel(n_classes, input_encoders)` | Per-attribute LSTM for temporal smoothing |
| `Qwen3VLClassifier(model_id, lora_r, ...)` | Qwen3-VL with LoRA for fine-tuning |

### Training

| Symbol | Description |
|--------|-------------|
| `irap_local_rec_trainer` | Supervised trainer (2 frozen + 8 finetune epochs, color jitter, dynamic weights) |
| `irap_local_rec_trainer_multiscale` | Supervised trainer with multi-scale augmentation |
| `irap_semisup_trainer` | Semi-supervised consistency regularization trainer |
| `irap_pseudo_label_trainer` | On-the-fly pseudo-label trainer |
| `irap_pseudo_label_offline_trainer` | Offline pseudo-label trainer |
| `combined_train_loader_f` | Training `data_loader_f` mixing separate `train*` splits at constant per-batch proportions (per-dataset `batch_size`) |
| `vlm_finetune_trainer` | VLM LoRA fine-tuning trainer |
| `FreezeThenFinetune` | Extension managing backbone freezing schedule |
| `MultiScaleSupervisedStep` | Multi-scale train step |
| `MultiAttributePseudoLabelStep(pre_trained_teacher, conf_thresh, temperature)` | Pseudo-label train step |

### Metrics & Loss

| Symbol | Description |
|--------|-------------|
| `get_irap_metrics(dataset, class_counts, attrs_to_include)` | Canonical metric factory |
| `MultiAttributeClassificationMetrics` | Per-attribute accuracy, precision, recall, F1, IoU |
| `MultiAttributeCrossEntropyLoss` | Per-attribute CE loss with optional class weighting |
| `DynamicBalancedRecallWeights` | Trainer extension for per-epoch class weight recomputation |

### Semi-supervised

| Symbol | Description |
|--------|-------------|
| `make_semisup_data(base_data, labeled_ratio, ...)` | Create labeled/unlabeled splits over a base dataset dict |
| `multi_attribute_kl_div_ll()` | KL divergence across attribute tuples |

### VLM

| Symbol | Description |
|--------|-------------|
| `vlm.Qwen3VLPredictor` | Zero-shot predictor (HuggingFace) |
| `vlm.Qwen3VLvLLMPredictor` | Zero-shot predictor (vLLM) |
| `vlm.PromptBuilder` | Prompt builder with configurable detail levels |
| `vlm.make_response_scheme(name)` | Response scheme factory |
| `make_vlm_bih_data()` | VLM fine-tuning dataset factory |
| `FineTunedVLMPredictor` | Inference from fine-tuned VLM checkpoints |

### Utilities

| Symbol | Description |
|--------|-------------|
| `export_feats(split, feat_dir)` | Export model features as `.npy` |
| `vistas_params_spec(...)` | Construct `--params` translation string for Vistas weights |
| `get_attrs_to_include()` | Canonical 41-attribute subset |
| `map_attr_names_to_indices(...)` | Map attribute names to dataset indices |

## Troubleshooting

- **Missing `seg_to_res/*.pickle`**:
  Either create the pickle files under `IRAP_BIH_METADATA/seg_to_res/`, or disable filtering with `make_bih_data(use_ncontext_filter=False)`.

- **`vistas.pt` not found**:
  Put it at `<VIDLU_PRETRAINED>/irap_gaim/vistas.pt`, or use an absolute path in the `--params` string.

- **VLM out of memory**:
  Use `load_in_4bit=True` (default) for `Qwen3VLClassifier`, or switch to `Qwen3VLvLLMPredictor` with vLLM's memory-efficient batching.

- **VLM response parsing failures**:
  Try a different `ResponseScheme` (e.g., `JsonResponseScheme` for more structured output) or increase the `detail_level` to give the model more context about valid values.


## References

[1] M. Kačan, M. Ševrović and S. Šegvić, "Dynamic Loss Balancing and Sequential Enhancement for Road-Safety Assessment and Traffic Scene Classification," in IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 15628-15640, Nov. 2024, doi: 10.1109/TITS.2024.3456214.