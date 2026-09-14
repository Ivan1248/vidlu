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
timm>=1.0.20, huggingface_hub>=0.30   # ViT backbones and their weight downloads
```

For VLM inference (optional):

```
transformers>=5.14, qwen-vl-utils>=0.0.8, pyyaml>=6.0
```

For VLM fine-tuning (optional): `peft`, `bitsandbytes`, `accelerate`. Qwen3.5 additionally
benefits from `causal-conv1d` and `flash-linear-attention` (fused Gated-DeltaNet kernels;
transformers falls back to a slower pure-torch path without them).

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

### Training-split sizes

With the default context sequence `(0, -1, -4)`:

| Release | Training examples |
|---|---|
| iRAP-BiH | 209,459 |
| iRAP-Vietnam | 10,818 |

A natural BiH:Vietnam ratio of **19.4:1**, which is what the `combined_train_loader_f`
mixing ratio should be read against (see
[Joint BiH + Vietnam training](#joint-bih--vietnam-training)), and which puts the two
releases in different regimes for choosing a fine-tuning strategy (see
[Encoders](#encoders)). These counts are worth stating explicitly because several
configuration choices only make sense relative to them.

### Inspecting the label distribution

Three metadata-only tools; none loads an image, and all run as plain files without
importing torch or vidlu.

```bash
# Per-split example counts.
IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/dataset_split_sizes.py

# Per-attribute class distributions: how much is labelled, how skewed, which classes are
# rare or never observed. Attributes a release does not annotate (7 of 41 on Vietnam) are
# named once and then excluded from every table and count.
IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/attribute_distribution_report.py \
  --releases bih vietnam --splits train val test --rare-fraction 0.01 -o dist.json

# Every class of every attribute as one figure: log axis, grouped by attribute, with
# never-observed classes marked rather than dropped.
IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/attribute_class_histogram.py \
  --release vietnam --split train -o vietnam_train_classes.png
```

`attribute_class_histogram.py` delegates its counting to `attribute_distribution_report`,
so the figure and the report cannot disagree. Options:

| Option | Effect |
|---|---|
| `--normalize` | Plot each class as a share of its own attribute, making attributes with different amounts of labelling comparable |
| `--linear` | Linear value axis. Honest about magnitude, but it hides the rare tail — hence the logarithmic default |
| `--sort-attributes {schema,count,imbalance}` | Order of attribute groups. `count` is by descending labelled examples; `imbalance` puts the most skewed attributes first |
| `--sort-classes {schema,count}` | Order within each group. `schema` keeps class-index order, which is meaningful for ordinal attributes such as speed limits; `count` is by descending examples, the same key one level down |

```bash
# Most skewed attributes first, classes by descending count within each.
IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/attribute_class_histogram.py \
  --release vietnam --sort-attributes imbalance --sort-classes count -o vietnam_skew.png
```

Classes with no examples are drawn at a floor in a distinct colour on both scales rather
than omitted: on a logarithmic axis zero cannot be placed at all, and on a linear one it
has zero width, so either way an unobserved class would look like a missing one.

### N-context filtering (default behavior)

`irap_gaim.make_bih_data()` applies an N-context filter by default (`use_ncontext_filter=True`) using precomputed pickle files:
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/train.pickle`
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/val.pickle`
- `$IRAP_HOME/IRAP_BIH_METADATA/seg_to_res/test.pickle`

To disable this filtering (use all segments that pass label/context checks):

```bash
python scripts/run.py train \
  "irap_gaim.make_bih_data(use_ncontext_filter=False)" \
  "id" \
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
│       ├── base.py              # FrameEncoder: the backbone contract
│       ├── resnet.py            # ResNetEncoder (ImageNet / Vistas)
│       ├── vit.py               # ViTEncoder, dinov2_vit_encoder, HF download helpers
│       ├── mae.py               # MAEEncoder, mae_vit_encoder
│       ├── timm_vit.py          # TimmViTEncoder: SigLIP 2, SPAR, DINOv3
│       ├── qwen_vision.py       # Qwen3VLVisionEncoder (LoRA)
│       └── attention.py         # Attention pooling
├── training/                    # Training infrastructure
│   ├── configs.py               # Trainer configurations (supervised, semi-sup, pseudo-label, VLM)
│   ├── optim.py                 # EncoderOptimizerMaker (trainability + layer-wise LR decay)
│   ├── steps.py                 # MultiScaleSupervisedStep, MultiAttributePseudoLabelStep
│   ├── extensions.py            # FreezeThenFinetune, MultiAttributeScorePrinter, VisualizationExtension
│   ├── dynamic_weights.py       # DynamicBalancedRecallWeights (per-epoch class reweighting)
│   ├── semisup.py               # Semi-supervised splits, pseudo-label generation, adaptive thresholds
│   └── jitter.py                # Color jitter augmentation
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
│       ├── model.py             # Qwen3VLClassifier / Qwen35Classifier / Gemma4VLClassifier (LoRA wrappers)
│       ├── dataset.py           # VLMIrapDataset, make_vlm_bih_data
│       ├── loading.py           # load_finetuned_classifier (a checkpoint without an experiment)
│       ├── predictor.py         # VLMClassifierPredictor, run_eval
│       └── steps.py             # VLMTrainStep, VLMEvalStep
├── tools/                       # Utility scripts and visualization
│   ├── vis_utils.py             # Visualization utilities (color palettes, composite images)
│   ├── dataset_viewer.py        # Streamlit interactive data browser
│   ├── inference.py             # Evaluation hook for structured predictions
│   ├── inference_visualization.py  # Standalone PNG generation
│   ├── generate_pseudo_labels.py   # Offline pseudo-label generation
│   ├── baseline_random.py       # Random baseline predictor
│   ├── attribute_most_common_report.py  # Attribute frequency analysis
│   ├── dataset_split_sizes.py   # Per-split example counts
│   ├── attribute_distribution_report.py  # Per-attribute class distributions
│   ├── attribute_class_histogram.py      # All class frequencies in one figure
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

**Important**: the dataset returns RGB in \([0,1]\) and **each encoder applies its own
normalization**, so the input-adapter positional is **`id` for every backbone**. The
statistics a backbone needs are a property of how it was pretrained, so they are read
from the checkpoint (timm's `resolve_model_data_config`, or Hugging Face's
`preprocessor_config.json`) rather than chosen on the command line. `ResNetEncoder` is
the exception, since it is trained here from ImageNet or Vistas initialization rather
than carrying its own statistics: it takes `pixel_stats=data.train.info.pixel_stats`
explicitly.

### ResNet encoder (Vistas-pretrained backbone)

Deterministic loading + center crop in the dataset, photometric jitter in the trainer.

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
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
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
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
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.dinov2_vit_encoder,variant='dinov2_vitb14',params_dir=dirs.pretrained)" \
  "irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

### MAE ViT encoder (ImageNet-pretrained)

Loads a Hugging Face `ViTMAEModel` (`facebook/vit-mae-base` by default) as a plain
ViT feature extractor — MAE random masking is disabled (`mask_ratio=0`). It applies
ImageNet normalization, read from the checkpoint's own `preprocessor_config.json`:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.mae_vit_encoder,variant='facebook/vit-mae-base',params_dir=dirs.pretrained)" \
  "irap_gaim.irap_local_rec_trainer_nofreeze" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

Use `variant='facebook/vit-mae-large'` / `'-huge'` (or any HF ViT-MAE repo id) for
larger models. `irap_local_rec_trainer` (freeze-then-finetune) also works, since the
encoder pools by CLS token and so has no `pool_parameters()` (the frozen phase trains
heads only).

### Qwen3-VL image encoder (LoRA fine-tuning)

Uses the **vision tower** of `Qwen/Qwen3-VL-8B-Instruct` as the backbone, fine-tuned
with LoRA (the base tower stays frozen). This is distinct from `Qwen3VLClassifier`,
which fine-tunes the whole generative VLM. The tower is run per frame, so the model's
own image processor handles patchification and Qwen normalization — feed raw `[0,1]`
inputs with the `id` adapter. Requires `peft` (and `bitsandbytes` for 4-bit):

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.Qwen3VLVisionEncoder,model_id='Qwen/Qwen3-VL-8B-Instruct',lora_r=16,load_in_4bit=True)" \
  "irap_gaim.irap_vit_lora" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

`irap_vit_lora` puts the encoder in `'lora'` trainability, so the optimizer receives the
heads and the adapters only. The encoder rejects `'all'` outright: its base tower is
loaded frozen (and optionally 4-bit quantized) and is never checkpointed, so training it
would produce weights a resumed run could not restore. Only `attention=False` is
supported for now, because the feature map needs a uniform patch grid across the batch
and Qwen patchifies at dynamic resolution.

### SigLIP 2, SPAR, and DINOv3 (timm backbones)

These three are the same timm vision transformer with different weights, so one encoder
class covers them. Each declares its own normalization and feature width, and frames are
encoded at their native 384×288 (an 18×24 patch grid) — `dynamic_img_size=True` resamples
the position embeddings, and DINOv3 uses RoPE and is resolution-agnostic by construction.

| Backbone | Factory | timm model | Dim | Normalization |
|---|---|---|---|---|
| SigLIP 2 ViT-B/16 @512 | `siglip2_vit_encoder(variant='base-512')` | `vit_base_patch16_siglip_512.v2_webli` | 768 | mean = std = 0.5 |
| SPAR ALL SigLIP2 ViT-B/16 | `siglip2_vit_encoder(variant='base-512', spar_params_path=...)` | same module | 768 | mean = std = 0.5 |
| DINOv3 ViT-L/16 | `dinov3_vit_encoder(variant='large')` | `vit_large_patch16_dinov3.lvd1689m` | 1024 | ImageNet |

Three fine-tuning regimes, one trainer config each — all backbone-agnostic, so only the
model string changes between them:

| Trainer | Regime | Optimizer |
|---|---|---|
| `irap_vit_linear_probe` | backbone frozen; heads (and a parametric pooling head) only | AdamW, lr 1e-3, cosine |
| `irap_vit_partial_finetune` | last `num_trainable_blocks` blocks + output norm + pooling head | AdamW, lr 1e-4, wd 0.05, `layer_decay` 0.75, heads ×10 |
| `irap_vit_finetune_llrd` | full fine-tune, **ViT-B** | AdamW, lr 1e-4, wd 0.05, `layer_decay` 0.65, heads ×10, warmup + cosine |
| `irap_vitl_finetune_llrd` | full fine-tune, **ViT-L** | same but `layer_decay` 0.75 |
| `irap_vit_lora` | LoRA adapters only (needs `lora_r=` on the encoder) | AdamW, lr 1e-4, cosine, **batch 4** |

All of these use `batch_size=12`, the same as the ResNet recipe, so switching backbones
changes one thing rather than two. A 3-frame example is 3 backbone images, so that is 36
images per step: roughly 4 GB for a ViT-B full fine-tune and 12 GB for ViT-L, or ~6 GB with
`grad_checkpointing=True` on the encoder. `irap_vit_lora` is the exception at batch 4,
because its original user is the 8B Qwen3-VL tower; a timm ViT under LoRA can override it
to 12.

Batch size and learning rate are coupled — MAE states its rate as `base_lr * batch/256` —
while the rates above are absolute, so scale the rate if you override `batch_size`.

`layer_decay` differs by depth because it compounds over blocks: the values are MAE's
(`--layer_decay 0.65` for ViT-B, `0.75` for ViT-L/H, `--weight_decay 0.05`; He et al.,
[arXiv:2111.06377](https://arxiv.org/abs/2111.06377), and the reference implementation's
[FINETUNE.md](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md)). At 0.65 a
24-block ViT-L would train its first block at ~3e-5 of the base rate. Each config has a
`*_nodyn` variant without `DynamicBalancedRecallWeights`.

#### Which regime to start with — measured on Vietnam (2026-09-02)

MAE's values come from ImageNet-1k fine-tuning (1.28M images), and iRAP-Vietnam's 10,818
training examples are ~0.8% of that, which argues for probing or partially fine-tuning a
300M-parameter ViT-L. **That argument does not survive contact with the data.** DINOv3
ViT-L, iRAP-Vietnam, `val/amF1` over the 34 scoreable attributes:

| Regime | Epochs | amF1 | Runs |
|---|---|---|---|
| `irap_vit_linear_probe` | 56 | 0.4189 | 1 |
| `irap_vit_partial_finetune` | 56 | 0.4334 | 2 (mean) |
| `irap_vitl_finetune_llrd` | 5–20 | 0.4542 | 8 (mean) |

Full fine-tuning wins monotonically, and it does so while the two cheaper regimes get the
*longer* schedule. For reference, the ResNet-18/Vistas recipe (`irap_local_rec_trainer`)
reaches 0.4022 on the same split, so DINOv3 ViT-L full fine-tuning is **+0.052 over the
pre-ViT baseline**.

**The noise floor is ~0.007 amF1.** Four configurations were run twice under different
experiment tags; the replicate spreads were 0.0072, 0.0064, 0.0063 and 0.0091. Treat any
difference below ~0.01 as unmeasured. In particular:

- The full-fine-tuning cluster (0.449–0.458 across `epoch_count` 5/10/20 and `attention`
  on/off) is **one result, not a ranking**.
- `attention=True` is indistinguishable from `attention=False` — 0.4576 vs 0.4574 and
  0.4512 vs 0.4502 in the two matched pairs — while costing parameters and compute. Leave
  it off.

`epoch_count` is the one axis worth reading closely, because the three metrics disagree.
From 5 to 20 epochs, amF1 is flat, `val/acc` rises slightly (0.806 → 0.813–0.823), and
`val/loss` is far worse for every longer run (1.52 at 5 epochs against 2.22–3.19 at 10 and
2.85 at 20). Longer training buys a little majority-class accuracy, no macro-F1, and much
worse calibration. So prefer **`epoch_count=5`**: cheaper, and the only sane choice if the
confidences are consumed downstream (thresholded pseudo-labels — see [semisup.md](semisup.md)).

Backbones at a matched regime (`irap_vit_partial_finetune`, 56 epochs, Vietnam):

| Backbone | amF1 |
|---|---|
| DINOv3 ViT-L | 0.4334 |
| DINOv3 ViT-B | 0.4305 |
| SigLIP 2 ViT-B | 0.4055 |
| SPAR ALL SigLIP 2 ViT-B | 0.3997 |

DINOv3 beats SigLIP 2 by ~0.028, comfortably above the noise floor; SPAR's distillation
does not help on this task; and **ViT-L is indistinguishable from ViT-B under partial
fine-tuning**. The extra capacity only pays under a full fine-tune, where DINOv3 ViT-L
reaches 0.4542 against SPAR-SigLIP 2 ViT-B's 0.4304 (and SPAR degrades to 0.4066 when its
schedule is stretched to 20 epochs, so it overfits faster than DINOv3 does).

**DINOv3 ViT-B under `irap_vit_finetune_llrd` has not been run**, and is the obvious
missing cell: if it matches ViT-L it is the same result at roughly a third of the compute
and without `grad_checkpointing`.

Qwen3-VL-8B's vision tower under 4-bit LoRA reaches 0.4316 over 40 epochs — level with
partially fine-tuned DINOv3, below full fine-tuning, at 8B parameters. Not a promising
direction on this data. (Those runs used `irap_local_rec_trainer_qwen_nodyn`, a
Qwen-specific unweighted variant that is **not in this checkout**; reproducing them needs
that config committed first.)

**On BiH none of this is measured.** The only BiH ViT run so far is a SigLIP 2 linear probe
(0.5425, not comparable to any Vietnam number — 41 attributes rather than 34). The a priori
argument that full fine-tuning is at least as safe on 209,459 examples still stands, and
the Vietnam result now supports it rather than opposing it.

Run `irap_vit_linear_probe` once per backbone regardless, though for a different reason:
it is the clean measurement of *backbone quality*, which is what the SPAR-vs-SigLIP 2 and
DINOv3 comparisons are actually asking. A full fine-tune on 209k examples can relearn
enough to wash out the differences between backbones.

Do the backbone comparison on **Vietnam first**: it is 19x smaller, it is the primary
target, and it is the setting where pretrained feature quality matters most. Confirm the
winner on BiH rather than sweeping there.

Note that heavy class imbalance is **not** an argument for freezing the backbone: the trunk
is shared across all 41 attributes and is trained by every example, so its effective sample
size is the whole split. Rare classes are a head-and-loss problem — that is what
`DynamicBalancedRecallWeights` addresses — and freezing the backbone does nothing for them.

Linear probing on SigLIP 2, on BiH:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.siglip2_vit_encoder,variant='base-512')" \
  "irap_gaim.irap_vit_linear_probe" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

DINOv3 ViT-L on Vietnam, probing the frozen backbone — the regime DINOv3 is designed and
evaluated for:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_vietnam_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.dinov3_vit_encoder,variant='large')" \
  "irap_gaim.irap_vit_linear_probe" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

For partial or full fine-tuning, swap in `irap_vit_partial_finetune` or
`irap_vitl_finetune_llrd` and add `grad_checkpointing=True` to the encoder — it trades
compute for activation memory, which is what makes three frames of a 24-block ViT-L fit.

DINOv3 weights are gated behind Meta's
[DINOv3 licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license):
accept it on the model page and export `HF_TOKEN`, or the download fails with HTTP 401.

#### DINOv3 specifics

- **It is meant to be frozen — but on iRAP it should not be.** The paper's claim is that
  "a single frozen SSL backbone can serve as a universal visual encoder that achieves
  state-of-the-art performance on challenging downstream tasks"
  ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)), and every downstream result in it
  uses a frozen backbone. It publishes **no full-fine-tuning recipe**, so
  `irap_vitl_finetune_llrd`'s values are MAE's, not DINOv3's. Transplanted recipe
  notwithstanding, full fine-tuning beats the frozen probe by 0.035 amF1 on Vietnam —
  ~5x the run-to-run spread — so the frozen-backbone claim does not transfer to iRAP
  attribute classification (see the regime table above).
- **Pooling.** `dinov3_vit_encoder` defaults to `pooling='cls+avg'` — the class token
  concatenated with the average patch token, giving a 2048-d feature. That is what
  DINOv3's own linear evaluation classifies (`dinov3/eval/linear.py`, `use_avgpool`).
  timm's default head averages the patch tokens alone and discards the class token, which
  is *not* the evaluated protocol; pass `pooling='default'` to get it back.
- **The probe learning rate wants sweeping.** DINOv3's linear evaluation does not fix one:
  it sweeps `1e-5 … 1e-1` (SGD, momentum 0.9, weight decay 0, batch 128, 10 epochs, cosine)
  and reports the best. `irap_vit_linear_probe`'s single `lr=1e-3` is the least-justified
  number in these configs. This is also the main caveat on the regime table above: the
  probe rung is a single run at a single unswept LR, so 0.4189 is a lower bound on what
  probing can do. The full-vs-probe gap (0.035, ~5x the noise floor) is large enough that
  it is unlikely to close entirely, but the margin is not established until the sweep runs.
- **Resolution.** DINOv3 ViTs use RoPE and have no position-embedding table, so 288×384 is
  handled natively with no interpolation.
- Register tokens: ViT-L/16 has a class token plus 4 registers; both are excluded from the
  spatial feature map.

#### SPAR

[SPAR](https://github.com/naomikombol/SPAR) (*Single-Pass Any-Resolution ViT for
Open-vocabulary Segmentation*, Kombol, Martinović, Šegvić & Tolias, CVPR 2026,
[arXiv:2604.02252](https://arxiv.org/abs/2604.02252)) distils a finely-strided
sliding-window teacher into a single-pass student, which makes the backbone much better
behaved away from its native resolution — the regime here, since frames are 384×288 and
the tower was pretrained at 512×512. **"ALL" means all parameters were trained** (SPAR's
default unfreezes only the last two blocks), distilled on 25k SA-1B images. The
architecture is unchanged, so it is a drop-in weight swap against stock SigLIP 2, which
makes the two a clean A/B.

The weights are published as Google Drive files rather than on a hub, so download the
"SPAR ALL SigLIP2" checkpoint from the SPAR README into `dirs.pretrained` and pass its
path. Only the ViT trunk is used; the open_clip wrapper and text tower are stripped, and
loading is strict, so a wrong extraction fails instead of leaving the backbone partly
random:

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_vietnam_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.siglip2_vit_encoder,variant='base-512',spar_params_path=dirs.pretrained/'SPAR_ALL_ViT-B-16-SigLIP2-512.pt')" \
  "irap_gaim.irap_vit_finetune_llrd" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

Note that SPAR's own published hyperparameters (AdamW 2e-5, constant LR, 10 epochs,
batch 1, MSE) are a *distillation* recipe and say nothing about classification
fine-tuning; the recipes above come from MAE/DINOv3 instead.

### iRAP-Vietnam

iRAP-Vietnam shares the same 41-attribute schema (same `attribute_metadata.json`)
as iRAP-BiH, but does not annotate 7 of the attributes (the flow attributes,
Upgrade cost, Roadworks, Bicycle facility); those columns are the ignore label
`-1` in every Vietnam target, so the loss skips them per example and the metrics
report `n=0` for them.

```bash
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "irap_gaim.make_vietnam_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(data.train)"
```

Pass the loaded training split to `get_irap_metrics(data.train)`; the no-argument
form silently reloads `make_bih_data()` to resolve the attribute set.

## Encoders

Every encoder implements `FrameEncoder` (`models/encoders/base.py`):

```python
forward(frames: (B, 3, H, W) in [0, 1]) -> (feature_map: (B, C, gh, gw), pooled: (B, D))
                               # widths are resolution-independent; the classifier's heads
                               # are sized from the output of its first call
pixel_stats                    # the normalization applied, from the checkpoint
set_trainable('none' | 'pool' | 'lora' | 'all')
param_groups(lr=, weight_decay=, layer_decay=)
```

The input-adapter positional is `id` for all of them.

| Encoder | Factory | Pretrained weights | `'pool'` trains | Notes |
|---------|---------|-------------------|-----------------|-------|
| ResNet-18 | `ResNetEncoder(pretrained=True, pixel_stats=...)` | ImageNet (torchvision) | SPP | Good baseline; takes the dataset's statistics |
| ResNet-18 + Vistas | `ResNetEncoder(pretrained=False, pixel_stats=...)` + `--params` | Vistas `.pt` file | SPP | Best for road scenes; load via `--params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt"` |
| DINOv2 ViT-B/14 | `dinov2_vit_encoder(variant='dinov2_vitb14')` | Auto-downloaded | — (CLS) | Self-supervised; no `--params` needed |
| MAE ViT-B/16 | `mae_vit_encoder(variant='facebook/vit-mae-base')` | Auto-downloaded (HF) | — (CLS) | Masking disabled |
| SigLIP 2 ViT-B/16 | `siglip2_vit_encoder(variant='base-512')` | Auto-downloaded (timm) | attention pooling | 768-d; mean = std = 0.5 |
| SPAR ALL SigLIP2 | `siglip2_vit_encoder(variant='base-512', spar_params_path=...)` | Manual (Google Drive) | attention pooling | Resolution-robust SigLIP 2; see [SPAR](#spar) |
| DINOv3 ViT-L/16 | `dinov3_vit_encoder(variant='large')` | Auto-downloaded (HF, **gated**) | — (CLS ⊕ average) | 2048-d pooled; needs `HF_TOKEN` |
| Qwen3-VL vision tower | `Qwen3VLVisionEncoder(model_id='Qwen/Qwen3-VL-8B-Instruct',lora_r=16)` | Auto-downloaded (HF) | — | LoRA only; use `irap_vit_lora`; needs `peft` |

## Metrics & dynamic weighting

### Metrics

`irap_gaim.get_irap_metrics(...)` builds one `vidlu.metrics.MultiAttributeClassificationMetrics` over the canonical 41-attribute subset: per attribute an `AttributeSpec(index, class_count)`, with the index taken from the schema order of `dataset.info.attr_to_value_to_class_idx`. Attributes the release does not annotate (via `info.attr_to_num_labeled`) are dropped, since they would otherwise score NaN from an empty confusion matrix. The metric itself is generic: any dataset with several categorical properties of one input can use it directly, e.g. `MultiAttributeClassificationMetrics({"gender": (0, 2), "age": (1, 5)}, metrics=("amF1", "aA", "_F1"))`. Accuracy is reported as `aA`, the mean over attributes of the per-attribute accuracies, which is the original `train_local_rec.py` definition (`accuracy_score` per attribute, then `np.mean` over the 41 attributes). The former `acc` pooled all labeled (segment, attribute) pairs instead; the two agree on BiH, where every segment is labeled for every attribute, and differ on IRAP-Vietnam, whose attributes have different label counts.

```bash
--metrics "irap_gaim.get_irap_metrics(data.train)"
```

### Support-restricted metrics (`_suppN`)

`amF₁` is the mean over attributes of each attribute's macro-F₁, so on a small split it
averages over many classes with single-digit support, whose F₁ is close to a coin flip. A
metric name may therefore carry a `_suppN` suffix — **restricted to classes with at least `N`
ground-truth examples in the evaluated split**.

The suffix is spelled out rather than borrowing the `@N` of Precision@k / AP@0.5, where `@`
denotes a cutoff on the *predictions* (rank, IoU) and would invite exactly the wrong
reading; `supp` rather than `sup` because `sup` reads as a supremum. The nearest
established convention is the many-/medium-/few-shot split of long-tailed recognition,
which buckets by *training* frequency; this thresholds on evaluation-split support instead,
for the reason in the first bullet below.

| Name | Meaning |
|---|---|
| `amF1_suppN` | `amF₁` over classes with support ≥ `N` (scalar, shown on the console) |
| `_mF1_suppN` | the same per attribute (`mF1_suppN` under a console-hidden key: a printer column, not on the log line) |
| `_nc_suppN` | how many classes entered that attribute's mean |

Thresholds default to `IRAP_CLASS_SUPPORT_THRESHOLDS = (5, 10)`; pass
`get_irap_metrics(data.val, min_class_supports=(5, 20))` to change them, or `()` for the
unrestricted metrics only. `_suppN` applies to the means over classes (`mP`, `mR`, `mF1`,
`mIoU`); `A_supp10` raises rather than being silently ignored.

Three things worth knowing:

- **Support is the confusion matrix's row sums — ground truth only, independent of the
  predictions.** The restricted class set is therefore identical for every model evaluated
  on a split, which is what makes `_suppN` values comparable across models. It *does* differ
  between splits, so do not compare `amF1_supp10` on validation against `amF1_supp10` on test.
- **`amF1` vs `amF1_supp1` isolates the absent-class penalty.** `IRAP_IGNORE_MISSING_CLASSES`
  masks with `actual + fp > 0`, so a class with no ground-truth example is re-admitted with
  F₁ = 0 as soon as the model predicts it once. `_supp1` uses ground-truth support only, so the
  difference between the two is exactly what those false positives cost. This matters
  because `DynamicBalancedRecallWeights` (below) actively pushes the model toward
  predicting rare classes.
- **An attribute with no qualifying class is dropped, not counted as zero.** It contributes
  NaN per attribute, and the average skips it; `_nc_suppN = 0` marks it. If no attribute
  qualifies, `amF1_suppN` is NaN.

Always read `amF1_suppN` next to `nc_suppN`: a higher restricted mean over fewer classes is not by
itself an improvement.

The `_suppN` names are understood by every `ClassificationMetrics` and
`ProbabilisticClassificationMetrics`, not only by the multi-attribute wrapper, so
`ClassificationMetrics(class_count, metrics=("mIoU", "mIoU_supp100", "nc_supp100"))` works for
plain semantic segmentation too. `get_irap_attribute_metrics` (sequential enhancement) does not
request them.

**What `_suppN` is and is not.** It is a variance-reduction device, not an imbalance metric:
it removes exactly the rare classes that imbalance work cares about, and because it thresholds
on evaluation-split support the class set is not the one any other paper reports. Read it as a
diagnostic beside `amF1`, never instead of it. The standard imbalance-aware summaries are
`amF1` (macro F₁) and `amR` (macro recall = balanced accuracy), both already reported, plus the
chance-corrected and probabilistic metrics below. The principled replacement, if `_suppN` is
ever retired, is the long-tailed-recognition convention of few/medium/many-shot buckets defined
by *training-split* frequency, which keeps the class set fixed across splits; the principled
remedy for low support itself is a confidence interval, not exclusion.

### Probability and chance-corrected metrics (NLL, Brier, mNLL, MCC)

Confusion-matrix metrics see only the arg-max. `get_irap_metrics` therefore also reports, per
attribute and averaged over attributes:

| Name | Meaning | Range |
|---|---|---|
| `aNLL` / `_NLL` | negative log-likelihood `-log p[target]`, mean over examples; the value the training loss optimizes, dominated by the tail | [0, ∞) |
| `aBrier` / `_Brier` | multi-class Brier score `Σ_k (p_k - 1[k = target])²` (Brier 1950; the sum form, not sklearn's binary one) | [0, 2] |
| `amNLL` / `_mNLL` | class-balanced NLL: per-class mean NLL, averaged over the classes with ground truth; weights every class equally, so a confidently wrong rare class costs as much as a common one | [0, ∞) |
| `amNLL_suppN` | the same restricted to classes with support ≥ `N`, like `amF1_suppN` | [0, ∞) |
| `aMCC` / `_MCC` | multi-class Matthews correlation coefficient (Gorodkin's R_k): one chance-corrected scalar per attribute, robust to skew | [-1, 1] |

Cohen's kappa is available by name (`akappa`, `_kappa`) but not in the default set. `MCC` is
**NaN, not 0**, when every ground-truth or every predicted label is one class (`kappa` when
both are): sklearn's 0 there is indistinguishable from chance level. A NaN attribute is dropped from the
attribute average, which is NaN only if no attribute has a defined value; the same rule makes
`get_irap_metrics` report NaN rather than `0.0` when no attribute contributed.

`NLL`/`Brier` need predicted distributions, and what an eval step puts in `out` differs, so
`get_irap_metrics` takes `output_kind`:

| `output_kind` | Produced by | Effect |
|---|---|---|
| `'logits'` (default) | `SupervisedStep` and every trainer built on it | log-softmax is applied |
| `'probs'` | `MultiScaleSupervisedStep` (`irap_local_rec_trainer_multiscale`) | the log is taken directly |
| `'hard'` | VLM text parsing (`vlm_inference`, `agent_classify`, VLM fine-tuning eval), `baseline_random` | `NLL`/`Brier`/`mNLL` are left out; requesting them raises |

Metrics come from `--metrics`, not from the trainer, so a mismatch cannot be detected: `'logits'`
on probabilities is silently wrong, and `'logits'` on one-hot outputs gives finite nonsense.
Zero probability on the true class is reported as an infinite `NLL` rather than clamped away.
`NLL` cannot be a `--main_metrics` choice yet because the checkpoint manager maximizes.

### Dynamic balanced recall weights

`DynamicBalancedRecallWeights` is a trainer extension that recomputes per-attribute class weights after each validation epoch. The weighting formula balances inverse class frequency with observed recall:

```
w = inv_freq * (1 - recall) + sqrt(inv_freq) * recall
```

This is configured automatically in the standard trainers. It requires a metric exposing `get_confusion_matrices()` (`MultiAttributeClassificationMetrics`) among the split's metrics; the per-class recalls are read from those matrices.

Which data each term comes from:

- **`inv_freq`** — class occurrence counts pooled over **every** `train*` split, so joint
  training gets the priors of the union rather than of whichever split comes first. The
  ignore label is excluded, not counted as a class. Attributes with no labelled training
  example (iRAP-Vietnam's seven BH-only ones) are left unweighted rather than given
  meaningless weights. The splits used are logged at startup.
- **`recall`** — the previous epoch's per-class validation recall, read from the metric
  belonging to the split that was just evaluated. By default only the **first** `val*`
  split drives the update; pass `recall_split_names=["val_vn", "val_bih"]` to pool several
  splits' confusion matrices instead.

Because the priors must describe the data the loss actually sees, a training split whose
`info` does not match it is rejected — see the note under
[Joint BiH + Vietnam training](#joint-bih--vietnam-training).

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
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
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
  "irap_gaim.make_semisup_data(irap_gaim.make_bih_data(), labeled_ratio=0.1)" "id" \
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
34 shared attributes are what matter in these combined-dataset experiments. The model
still has 41 heads either way, since `class_counts` is the full schema in both infos.

**Joining is incompatible with `DynamicBalancedRecallWeights`.** The joined split keeps
only one release's `info`, so `info.segment_ids` describes one part of the data while the
loss runs on both; class priors read from it would cover the wrong half. The extension
detects this (`len(info.segment_ids) != len(dataset)`) and raises rather than training on
silently wrong weights, so the concatenated recipe below wraps its trainer in
`without_dynamic_weights`. The balanced-loader recipe passes the splits separately and
needs no such workaround — the extension pools their counts.

Evaluation metrics are chosen **per split** (see
[Per-split evaluation metrics](#per-split-evaluation-metrics)): pass a
`dict(split_name=..., ...)` to `--metrics` to score `val_vn` on the 34 shared
attributes (NaN-free) and `val_bih` on all 41. The first `val*` split in the `data`
dict drives checkpoint selection, so list the Vietnam split first.

### Supervised, concatenated (proportions ∝ split sizes)

```bash
IRAP_HOME=~/data/datasets/ python scripts/run.py train \
  "b = irap_gaim.make_bih_data(); v = irap_gaim.make_vietnam_data(); data = dict(train=b.train.join(v.train, info=v.train.info), val_vn=v.val, val_bih=b.val)" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
  "irap_gaim.without_dynamic_weights(irap_gaim.irap_local_rec_trainer)" \
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
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train_vn.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train_vn.info.pixel_stats)" \
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
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
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

For multi-scale training and evaluation, use the `irap_local_rec_trainer_multiscale` trainer. Its `train_step` is a `MultiScaleSupervisedStep` and its `eval_step` is `None`, so the trainer derives the evaluation step from the training one by copying it with `eval=True` (`Trainer._get_eval_step`); the two therefore average over the same scales by construction. `eval` selects the role exactly as in `SupervisedStep`: evaluation runs without gradients and takes no optimization step.

The step puts the scale-averaged *probabilities* in `out`, so pass `--metrics "irap_gaim.get_irap_metrics(data.train, output_kind='probs')"`; with the default `'logits'` the `NLL`/`Brier` values would be silently wrong (see [Probability and chance-corrected metrics](#probability-and-chance-corrected-metrics-nll-brier-mnll-mcc)).

Training this way backpropagates through one forward pass per scale, so a step costs about `len(scales)` times a single-scale one in time and activation memory. The config keeps `batch_size=12`; lower it if the run does not fit.

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
# result.sessions:    list[VLMSessionResult] — one per (image, prompt) exchange
```

### Attributes per session

A *session* is one independent (image, prompt) -> response exchange.
`attrs_per_session` decides how the attributes are divided among them, and is the
one knob that separates the two evaluation regimes:

| `attrs_per_session` | Prompt | Sessions per image |
|---|---|---|
| `None` (default) | lists every attribute; the model classifies all of them at once | 1 |
| `1` | asks about one attribute | one per attribute |
| `n` | asks about `n` attributes | `ceil(A / n)` |

Fine-tuning always trains against `None` (`VLMIrapDataset` builds one prompt covering
every attribute), so evaluating with `1` deliberately puts a fine-tuned model off its
training distribution — that is the effect being measured, and
`tools/compare_session_modes.py` reports the format-error rates that separate it from a
genuine change in classification skill.

Two things adapt to the session size rather than being fixed:

- **The response instructions.** At one attribute a scheme returns its
  `SINGLE_ATTRIBUTE_RESPONSE_INSTRUCTIONS`: the same `N: VALUE` syntax and the same
  parser, but the wording stops describing a list and drops the example citing
  attribute numbers the prompt does not contain.
- **The response-token budget.** `max_response_tokens=None` derives it per session from
  `ResponseScheme.compute_max_response_tokens`, so it scales with how many attributes
  the session asks about. `response_token_margin` (default 32) is added on top: the bound
  exists to catch gross non-compliance, and a budget that *binds* in one arm of a
  comparison manufactures a difference between the arms. Always check the reported
  `truncation_rate` is ~0 before comparing.

**Batching.** Generation batches along the axis that holds the prompt fixed and varies
the image. The chat layout is text-then-image, so a fixed prompt is a shared prefix
(which vLLM's prefix caching reuses), every row of the batch comes out the same length
so no padding is needed, and no image is encoded twice. Batching over *attributes*
instead would re-encode one image once per attribute and produce ragged rows.
`batch_size` defaults to being sized from `batch_tokens` (16384), because prefill
saturates at a roughly constant token count — a fixed image count would saturate the
one-attribute arm and starve the all-attribute one, or vice versa.

Batching is verified in `tests/test_vlm_batched_generation.py`. Two of those tests need
real weights and are opt-in:

```bash
VLM_TEST_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct python -m pytest \
  vidlu_irap_gaim/tests/test_vlm_batched_generation.py -v
```

They check that a *padded* batch gives each row the same next-token logits as running
that row alone (Qwen3-VL derives M-RoPE position ids from the attention mask, so wrong
padding degrades rows silently rather than raising), and that batched and single-row
generation parse to the same class index on a constrained prompt. Note what is
deliberately *not* asserted: byte-identical free-form text. Batched matmuls reduce in a
different order, so logits differ in their last bits and an open-ended continuation can
diverge at a near-tied token — that is floating-point noise, not a defect, and the
quantity the metrics actually consume is the parsed class.

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

`Qwen35Classifier` fine-tunes `Qwen/Qwen3.5-9B`. Qwen3.5 shares Qwen3-VL's processor
(`AutoProcessor` resolves both to `Qwen3VLProcessor`), so it subclasses `Qwen3VLClassifier`
and inherits its tokenization and generation path; it only overrides the load class
(`AutoModelForMultimodalLM`), the attention preference (SDPA — `head_dim=256` trips many
flash-attn builds at forward), the LoRA targets, and thinking mode.

```bash
python scripts/run.py train "irap_gaim.make_vlm_bih_data()" "standardize" "irap_gaim.Qwen35Classifier,lora_r=64" "irap_gaim.qwen35_vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics(output_kind='hard')"
```

Two things are specific to this architecture:

- **Thinking mode is on by default** in the checkpoint's chat template. The class sets
  `_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`; without it the generation prompt
  ends on an open `<think>` block and the model emits a stray `</think>` ahead of the
  response, which makes response parsing fail silently (the eval step swallows parse errors
  into empty predictions, so metrics collapse to ~0 with only a warning).
- **LoRA targets cover three module families.** Only 8 of the 32 text layers are
  Gated-Attention (`q/k/v/o_proj`); the other 24 are Gated-DeltaNet
  (`linear_attn.in_proj_{qkv,z,b,a}`, `out_proj` — note PEFT's suffix matching means
  `o_proj` does *not* match `out_proj`), and every layer has a dense `mlp` FFN. The
  checkpoint is dense, not MoE, so 4-bit QLoRA works normally.

Requires transformers ≥ 5.x. `causal-conv1d` and `flash-linear-attention` are optional;
without them the Gated-DeltaNet layers fall back to a correct but slower pure-torch path.

### Custom prompts on a fine-tuned checkpoint

Training saves adapter-only checkpoints under the experiment directory. Both are printed by
`run.py`, given the same four positional argument strings the training command used (`--metrics`,
`-r` and the like do not affect the path):

```bash
python scripts/run.py get_checkpoint_path <data> <input_adapter> <model> <trainer> [--which best|last]
python scripts/run.py get_path <data> <input_adapter> <model> <trainer>   # the experiment directory
```

`get_checkpoint_path` prints one checkpoint directory (by default the best-performing one), which
is what the loader and the tools below take. Other initialization output precedes it, so pipe
through `tail -1` when capturing it:

```bash
CKPT=$(python scripts/run.py get_checkpoint_path <data> <input_adapter> <model> <trainer> | tail -1)
```

`load_finetuned_classifier` reconstructs the classifier from a checkpoint alone – the base
model, the LoRA configuration and the classifier class are stored in it, so the arguments of the
training command do not have to be repeated, and no dataset, trainer or metrics are built:

```python
from vidlu_irap_gaim import load_finetuned_classifier

model = load_finetuned_classifier(experiment_dir, which="best")  # or which="last"
response, thinking, is_truncated = model.generate_for_eval(image=pil_image, prompt="...")
```

The same thing from the command line, either once or as a prompt loop that keeps the model
loaded (loading a 4-bit Qwen3-VL takes minutes, so the loop is worth it for exploration):

```bash
python -m vidlu_irap_gaim.tools.vlm_prompt --checkpoint <experiment-dir>     --image frame.jpg --prompt "Describe the roadside hazards."
python -m vidlu_irap_gaim.tools.vlm_prompt --checkpoint <experiment-dir> --image frame.jpg
```

And in the browser (Streamlit ≥ 1.43):

```bash
streamlit run vidlu_irap_gaim/tools/vlm_chat_app.py -- --checkpoint <checkpoint-dir>
```

The page is served on port 8501. [`scripts/podman.sh`](../scripts/podman.sh) runs the container
with `--network=host`, so the page is reachable at `localhost:8501` on the host directly; with
plain docker/podman, pass `-p 8501:8501` (or `--network=host`):

```bash
bash scripts/podman.sh streamlit run vidlu_irap_gaim/tools/vlm_chat_app.py -- --checkpoint <checkpoint-dir>
```

From another machine, add `ssh -L 8501:localhost:8501 <host>`.

The same tools run the **pretrained model without the fine-tuning**, for comparison — no adapter
is attached at all, and everything else (prompting, image preprocessing, generation) is identical:

```python
from vidlu_irap_gaim import load_base_classifier

model = load_base_classifier()  # or load_base_classifier("Qwen/Qwen3-VL-8B-Instruct")
```

```bash
python -m vidlu_irap_gaim.tools.vlm_prompt --base-model --image frame.jpg --prompt "..."
streamlit run vidlu_irap_gaim/tools/vlm_chat_app.py -- --base-model
```

The base weights are 4-bit quantized by default, as they were during QLoRA training, so the two
runs differ only in the adapter; `--load-in-4bit`/`--no-load-in-4bit` overrides that.

The same tools run the **pretrained model without the fine-tuning**, for comparison. No adapter
is attached at all, and the prompting, image preprocessing and generation are identical, so the
runs differ only in the fine-tuning:

```python
from vidlu_irap_gaim import load_base_classifier

model = load_base_classifier()  # or load_base_classifier("Qwen/Qwen3-VL-8B-Instruct")
```

```bash
python -m vidlu_irap_gaim.tools.vlm_prompt --base-model --image frame.jpg --prompt "..."
streamlit run vidlu_irap_gaim/tools/vlm_chat_app.py -- --base-model
```

The base weights are 4-bit quantized by default, as they were during QLoRA training;
`--no-load-in-4bit` loads them unquantized. In the browser, the sidebar switches between the two
sources without restarting.

Responses are capped at 512 tokens by default, and a cut-off response is reported as such. To let one
run to its natural end, tick "No response-token limit" in the sidebar, pass `--no-response-limit`
to the command-line tool, or type `:tokens none` in its prompt loop; the response is then bounded
only by the model's context window.

Checkpoints written before the classifier class was recorded in the checkpoint are identified by
their base model instead, which covers every run that left `model_id` to the class. Only a run
that predates the recording *and* overrode `model_id` needs `--classifier-class`
(`classifier_class=` in Python).

Two caveats apply to all three:

- **Every question gets its own request.** `generate_for_eval` builds a single-turn
  conversation (one image, one question), so the transcript in the chat page is history for
  the reader, not context for the model.
- **The responses are free-form text.** They are not parsed into attribute predictions; for that,
  use `tools/vlm_inference.py`, which runs the response scheme's parser over dataset splits.
  A fine-tuned model is also not a general-purpose assistant – it saw one fixed prompt during
  training, so responses to unrelated prompts drift towards the trained response format.

### VLM tools

- **`tools/vlm_inference.py`** – the evaluation pipeline: sessions, batched generation,
  parsing, metrics, and per-(segment, attribute) records
- **`scripts/eval_vlm_checkpoint.py`** – run it on a checkpoint (or the pretrained model)
  outside any Vidlu experiment
- **`tools/compare_session_modes.py`** – compare runs that differ in `attrs_per_session`
- **`tools/vlm_benchmark.py`** – benchmarking script (prefix caching, throughput)
- **`tools/vlm_prompt.py`** – custom prompts on a fine-tuned checkpoint (single-shot or interactive)
- **`tools/vlm_chat_app.py`** – the same in the browser (Streamlit)

#### Comparing session granularities

Four runs: the pretrained model and the fine-tuned one, each evaluated with all
attributes in one session and with one session per attribute. Everything else is held
constant — same datasets, same prompt YAML, same `detail_level`, same response scheme,
greedy decoding, same metrics — and both models load through
`load_base_classifier` / `load_finetuned_classifier`, so they share a backend,
quantization and generation path and differ only in the LoRA adapter.

```bash
DATA="irap_gaim.make_vlm_bih_data(detail_level='attr_desc_vals', response_scheme='standard')"
MODEL="irap_gaim.Qwen3VLClassifier,lora_r=64"
TRAINER="irap_gaim.vlm_finetune_trainer,batch_size=2,eval_batch_size=4"
PRED=vidlu_irap_gaim.vlm.finetuning.predictor

# Pretrained (no -r: there is no checkpoint to load)
python scripts/run.py test "$DATA" id "$MODEL" "$TRAINER" \
  -m "$PRED:run_zero_shot_eval,e,attrs_per_session=None,output_subdir='sess/zs_all'"
python scripts/run.py test "$DATA" id "$MODEL" "$TRAINER" \
  -m "$PRED:run_zero_shot_eval,e,attrs_per_session=1,output_subdir='sess/zs_per'"

# Fine-tuned (-r best loads the checkpoint)
python scripts/run.py test "$DATA" id "$MODEL" "$TRAINER" -r best \
  -m "$PRED:run_full_eval,e,attrs_per_session=None,output_subdir='sess/ft_all'"
python scripts/run.py test "$DATA" id "$MODEL" "$TRAINER" -r best \
  -m "$PRED:run_full_eval,e,attrs_per_session=1,output_subdir='sess/ft_per'"

python -m vidlu_irap_gaim.tools.compare_session_modes \
  zs_all=<exp>/sess/zs_all/test/records.jsonl zs_per=<exp>/sess/zs_per/test/records.jsonl \
  ft_all=<exp>/sess/ft_all/test/records.jsonl ft_per=<exp>/sess/ft_per/test/records.jsonl \
  --baseline zs_all --output report.md
```

Each run writes `records.jsonl` (one row per segment and attribute: target, prediction,
validity, the raw response, its line count and token count, truncation) alongside
`predictions.json` and `summary.json`. The comparison is paired over segments — a
bootstrap CI on ΔamF1 resampling segments, and per-attribute McNemar tests with
Benjamini-Hochberg correction — plus the diagnostics that say *why* a number moved:
invalid rate, truncation rate, multi-line-response rate (a model asked about one
attribute that emits several lines is using the other format), and the slope of
correctness on the attribute's position in a multi-attribute prompt.

Every evaluation reports two scorings of an unusable response, as `metrics` and
`metrics_excluding_invalid` in `summary.json`: `class0` (matching the training-time
eval, but generous, since class 0 is often the majority value) and `ignore` (excluded,
measuring skill conditional on a format-compliant response). Neither subsumes the other, so
there is nothing to choose between -- both are always computed, and both are to be read
together with the invalid rate. The agent evaluator (`agent_classify evaluate`) reports
the same pair, so its runs and VLM runs compare directly.

## Inference & visualization

### Evaluation on standard splits

```bash
VIDLU_DETAILED_EVAL=1 IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py test \
  "irap_gaim.make_bih_data()" "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
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
  "irap_gaim.make_bih_data()" "id" \
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

### Single-command pipeline

`irap_gaim.train_seq_enh` runs the whole sequential-enhancement stage in one command
on top of a trained base experiment: it extracts and caches per-segment features for
all splits, trains one `GeneralLSTMModel` per attribute as a regular vidlu
experiment, and finally averages the per-attribute results into the base run's
multi-attribute metrics (`amF1`, `amP`, `amR`). Use the same data/model/trainer
strings (and the same `-e` suffix) as for base training, with `test`, a `-r` restore,
and `-m`:

```bash
python scripts/run.py test \
  "irap_gaim.make_vietnam_data()" id \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.Qwen3VLVisionEncoder,model_id='Qwen/Qwen3-VL-8B-Instruct',lora_r=16,load_in_4bit=True)" \
  "irap_gaim.irap_vit_lora_nodyn,epoch_count=20,eval_count=20" \
  -e 0 -r \
  -m "irap_gaim:train_seq_enh,e"
```

`-r` restores the last checkpoint; use `-r best` for the best one. A restore is
required — features extracted from untrained weights are useless, so the pipeline
refuses to run without one.

One forward pass writes both of the base model's per-segment outputs, under
`dirs.cache/seq_enh_feats/<key>/` (`<key>` identifies the base experiment + checkpoint;
see `provenance.json` there): its features to `feats/`, and its per-attribute logits to
`logits/`, with `attribute_slices.json` recording which columns belong to which
attribute. Reruns skip a segment only when *both* are present, so a cache written before
the logits were exported completes itself rather than being reused half-populated. Each
directory is then packed into `_packed_feats.npy` + `_packed_feats_index.json` and read
from there (memory-mapped), which is what keeps per-attribute training from re-reading
one file per sequence position; the per-segment `.npy` files remain the unit of
extraction and a pack is rebuilt whenever it no longer covers them.

The sequence carries only base-model *outputs* — features, logits, or the predicted
class — which is what this stage refines; ground truth enters solely as the target, for
the classified segment. `input_types=("feats","logits","labels")` selects which. The
window is `context_offsets`, which must contain 0 and be strictly increasing, and
defaults to a symmetric radius-10 window around the classified segment; every segment of
a window must lie in the same split, so a val/test segment is never refined using the
base model's predictions on its training neighbours.

Useful keyword arguments (append to the `-m` string): `attributes=[0,2]` restricts the
attribute set, `lstm_trainer=...` / `lstm_metrics=...` / `lstm_main_metric=...` override
the LSTM experiment configuration, and `lstm_tracker='wandb'` enables tracking for the
LSTM runs. The metric defaults come from `irap_gaim.metrics`: `DEFAULT_LSTM_METRICS`
uses `get_irap_attribute_metrics`, the single-attribute restriction of
`get_irap_metrics`, so `mF1`/`mP`/`mR`/`n` mean exactly what `amF1`/`amP`/`amR`/`_n`
mean in the base run; and `IRAP_MAIN_METRIC` (`mF1`) selects each attribute's best
checkpoint, matching the base run's `amF1`-based selection instead of accuracy.

Metadata paths are resolved from the base dataset automatically (works for both BiH and
Vietnam), and attributes no segment is labeled with (IRAP-Vietnam's five flow
attributes) are skipped. Each per-attribute LSTM experiment is independently
reproducible with the `run.py train` command printed when it starts, and the metrics it
reports come from the checkpoint that `lstm_resume` designates (`"best"` → the best one,
otherwise the last), not from whatever epoch training happened to end on.

### Manual feature export

Export per-segment features for a single split (the pipeline above does this
automatically; note the explicit `e` argument):

```bash
python scripts/run.py test \
  "irap_gaim.make_bih_data()" "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=data.train.info.class_counts,attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False,pixel_stats=data.train.info.pixel_stats)" \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim:export_feats,e,split='val',feat_dir='FEATS/val'"
```

The model must support `forward(..., return_features=True)` (the provided `ImageSequenceClassifier` does). Files already present in `feat_dir` are not recomputed.

### Manual sequential enhancement (LSTM smoothing)

`irap_gaim.make_seq_enh_data(...)` builds a per-attribute sequence dataset from the exported `.npy` outputs and exposes `class_count`, `feat_dim` and `target_index` on `info`. `irap_gaim.GeneralLSTMModel` expects `input_encoders`, not a raw `input_dim`, and reads its local context at `middle_index` — pass `info.target_index` rather than relying on the sequence-midpoint default, which is the classified segment only for a symmetric window:

```bash
python scripts/run.py train \
  "irap_gaim.make_seq_enh_data(feat_dir='FEATS/feats',attribute=0)" \
  "id" \
  "irap_gaim.GeneralLSTMModel,n_classes=data.train.info.class_count,middle_index=data.train.info.target_index,input_encoders=dict(feats=irap_gaim.IdentityEncoder(input_dim=data.train.info.feat_dim))" \
  "irap_gaim.seq_enh_lstm_trainer"
```

To use the base model's predictions as well, pass `logit_dir='FEATS/logits'` and
`input_types=('feats','logits','labels')`, adding
`logits=irap_gaim.IdentityEncoder(input_dim=data.train.info.class_count)` and
`labels=irap_gaim.LabelEmbeddingEncoder(num_embeddings=data.train.info.class_count)` to
`input_encoders`. More than one input type makes each example's `data` a mapping keyed by
input type instead of a bare tensor.

Accuracy is a default metric for the classification problem, so `--metrics` is not
needed. For numbers comparable to a multi-attribute run, add the iRAP metrics and select
checkpoints the same way:
`--metrics "irap_gaim.get_irap_attribute_metrics(data.train.info.class_count)" --main_metrics mF1`.

For releases that do not use the default `$IRAP_HOME/IRAP_BIH_METADATA` (e.g. IRAP-Vietnam, which colocates metadata with the dataset), pass `metadata_dir=...` to `make_seq_enh_data`.

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
| `make_seq_enh_data(feat_dir, attribute, logit_dir=, input_types=)` | Sequential enhancement dataset over the base model's per-segment outputs |
| `train_seq_enh(exp, ...)` | Single-command sequential-enhancement pipeline (feature caching + per-attribute LSTM training + `amF1` summary) |
| `pack_features(feat_dir)` | Packs per-segment `.npy` features into one memory-mapped array + index |

### Models

| Symbol | Description |
|--------|-------------|
| `ImageSequenceClassifier(class_counts, sequence_length, attention, encoder_f)` | Temporal sequence classifier; `set_encoder_trainable(mode)` selects the fine-tuning regime |
| `FrameEncoder` | The backbone contract: `(feature_map, pooled)`, declared dims, own normalization, `set_trainable`, `param_groups` |
| `ResNetEncoder(pixel_stats, pretrained=True\|False)` | ResNet-18/34/50 backbone; takes the dataset's pixel statistics |
| `dinov2_vit_encoder(variant, params_dir)` | DINOv2 ViT encoder factory |
| `mae_vit_encoder(variant, params_dir)` | MAE ViT encoder factory (masking disabled) |
| `TimmViTEncoder(model_name, pretrained, params_path, input_size, grad_checkpointing, lora_r)` | Any timm ViT as a backbone; supports layer-wise LR decay |
| `siglip2_vit_encoder(variant, spar_params_path)` | SigLIP 2 vision tower, optionally with SPAR weights |
| `dinov3_vit_encoder(variant)` | DINOv3 vision transformer (gated weights; needs `HF_TOKEN`) |
| `load_spar_trunk_state_dict(path)` | Extract the timm ViT trunk from a SPAR checkpoint |
| `Qwen3VLVisionEncoder(model_id, lora_r, load_in_4bit, ...)` | Qwen3-VL vision tower backbone with LoRA |
| `MultiScaleSequenceInference(base_model, scales)` | Multi-scale probability averaging wrapper |
| `GeneralLSTMModel(n_classes, input_encoders)` | Per-attribute LSTM for temporal smoothing |
| `Qwen3VLClassifier(model_id, lora_r, ...)` | Qwen3-VL with LoRA for fine-tuning |
| `Qwen35Classifier(model_id, lora_r, ...)` | Qwen3.5-9B with LoRA for fine-tuning (subclasses `Qwen3VLClassifier`) |
| `Gemma4VLClassifier(model_id, lora_r, ...)` | Gemma 4 multimodal (MoE) with LoRA for fine-tuning |

### Training

| Symbol | Description |
|--------|-------------|
| `irap_local_rec_trainer` | Supervised trainer (2 frozen + 8 finetune epochs, color jitter, dynamic weights) |
| `irap_local_rec_trainer_multiscale` | Supervised trainer averaging predictions over scales, in training and evaluation alike |
| `irap_vit_linear_probe` | Frozen backbone; trains the heads and any parametric pooling head |
| `irap_vit_partial_finetune` | Trains the last `num_trainable_blocks` blocks, the output norm and the pooling head |
| `irap_vit_finetune_llrd` | Full fine-tuning of a **ViT-B** backbone with layer-wise LR decay 0.65 |
| `irap_vitl_finetune_llrd` | Full fine-tuning of a **ViT-L** backbone with layer-wise LR decay 0.75 |
| `irap_vit_lora` | Trains injected LoRA adapters only (`Qwen3VLVisionEncoder`, or `TimmViTEncoder(lora_r=...)`) |
| `without_dynamic_weights(config)` | Derive a `*_nodyn` variant, dropping `DynamicBalancedRecallWeights` |
| `EncoderOptimizerMaker(optimizer_f, trainable, lr, layer_decay, head_lr_multiplier)` | Sets the encoder's trainability and builds its parameter groups |
| `irap_semisup_trainer` | Semi-supervised consistency regularization trainer |
| `irap_pseudo_label_trainer` | On-the-fly pseudo-label trainer |
| `irap_pseudo_label_offline_trainer` | Offline pseudo-label trainer |
| `combined_train_loader_f` | Training `data_loader_f` mixing separate `train*` splits at constant per-batch proportions (per-dataset `batch_size`) |
| `vlm_finetune_trainer` | VLM LoRA fine-tuning trainer |
| `gemma4_vlm_finetune_trainer` | VLM LoRA trainer tuned for Gemma 4 26B-A4B (bf16, naive MP across 4 GPUs) |
| `qwen35_vlm_finetune_trainer` | VLM LoRA trainer for Qwen3.5-9B (4-bit on one GPU, larger eval batch) |
| `FreezeThenFinetune` | Extension managing the two-phase freezing schedule; the phases' trainability is configurable (`frozen_trainability`, `finetune_trainability`) |
| `MultiScaleSupervisedStep` | Multi-scale train step |
| `MultiAttributePseudoLabelStep(pre_trained_teacher, conf_thresh, temperature)` | Pseudo-label train step |

### Metrics & Loss

| Symbol | Description |
|--------|-------------|
| `get_irap_metrics(dataset, class_counts, attrs_to_include, min_class_supports, output_kind)` | Canonical metric factory: one `MultiAttributeClassificationMetrics` requesting `irap_metric_names(...)` (`amF1`, `amP`, `amR`, `aA`, `aMCC`, plus `amF1_suppN` / `_mF1_suppN` / `_nc_suppN` per threshold; with `output_kind` `'logits'`/`'probs'` also `aNLL`, `aBrier`, `amNLL`, `amNLL_suppN`) |
| `irap_metric_names(min_class_supports, output_kind)` | The requested metrics as result key → metric name, console scalars first; per-attribute metrics get `vidlu.experiments.console_hidden` keys (`_mF1`, `_n`, ...), which `report_metrics` leaves off its log line and the tracker and `MultiAttributeScorePrinter` show without the prefix. The metric classes themselves do not know the convention. |
| `get_irap_attribute_metrics(class_count)` | The same protocol for one attribute (`mF1`, `mP`, `mR`, `n`) — used by sequential enhancement; no `_suppN` or probabilistic variants |
| `IRAP_ATTRIBUTE_METRIC_NAMES`, `IRAP_MAIN_METRIC` | The metric set and the checkpoint-selection metric, defined once for both |
| `IRAP_CLASS_SUPPORT_THRESHOLDS` | Default support thresholds `(5, 10)` for the `_suppN` variants |
| `vidlu.metrics.MultiAttributeClassificationMetrics(attributes, metrics, output_kind, ...)` | The generic metric: `attributes` maps a key to `AttributeSpec(index, class_count)`; per attribute a `ClassificationMetrics` (accuracy, P/R/F1/IoU, MCC, kappa) and, when requested, a `ProbabilisticClassificationMetrics` (NLL, Brier, class-balanced means); `get_confusion_matrices()` feeds `DynamicBalancedRecallWeights` |
| `vidlu.metrics.parse_metric_name`, `macro_over_supported`, `select_metrics` | The `[_][a]<base>[_suppN]` name grammar and the support-restricted mean, shared by all metric classes |
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
| `VLMClassifierPredictor` | Evaluation of a loaded classifier, pretrained or fine-tuned |
| `load_finetuned_classifier(checkpoint)` | Load a fine-tuned classifier from a checkpoint, without building an experiment |

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