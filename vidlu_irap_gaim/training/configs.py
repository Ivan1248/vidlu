from functools import partial, wraps

import torch

from vidlu.configs.training import TrainerConfig
from vidlu.optim.lr_schedulers import CosineLR, WarmupCosineLR
from vidlu.training.steps import SupervisedStep, SemisupConsStep
from vidlu.training.extensions import SemisupVAT
from vidlu.training.trainers import Trainer
from vidlu.configs.robustness import ph20_attack, ph3_attack

from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from .dynamic_weights import DynamicBalancedRecallWeights
from .extensions import (
    FreezeThenFinetune,
    MultiAttributeScorePrinter,
    VisualizationExtension,
)
from .jitter import make_sequence_color_jitter, JITTER_STRONG
from .optim import EncoderOptimizerMaker
from .steps import (
    MultiScaleSupervisedStep,
    MultiAttributePseudoLabelStep,
    ColorJitterAttack,
)

# `__all__` is computed at the bottom of this module (after all configs are
# defined): it is every module-level `TrainerConfig` plus the data-loader factory,
# so adding or removing a config needs no manual bookkeeping here.


# Training data-loader factory for joint training on datasets passed as separate `train*`
# splits: each batch gets a constant count from each (per-dataset `batch_size`, e.g. `[8, 4]`),
# and `primary_index='longest'` makes an epoch cover the larger dataset once while the smaller
# repeats in full shuffled passes. Only the mixing differs from `Trainer.data_loader_f`, so the
# rest (`dl_f`, `num_workers`, `shuffle`) is inherited from it rather than repeated. Evaluation
# is unaffected: a single split bypasses the multi-loader path.
combined_train_loader_f = partial(Trainer.data_loader_f, multi_dl_f="combine",
                                  primary_index="longest")


# Basic classification trainer with supervised step
# Loss is supplied externally as multi-attribute wrapper (see factory usage)
# Default epoch counts match the original repo's train_local_rec_paper_ep10.sh variant
# (2 frozen + 8 finetune = 10); the paper recipe (train_local_rec_paper.sh) is 2 + 13 = 15.
epoch_count = 2 + 8
irap_local_rec_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=False),
    train_step=SupervisedStep(amp=True),
    loss=MultiAttributeCrossEntropyLoss(),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,  # evaluation every epoch, which is required for DynamicBalancedRecallWeights to update class weights after each epoch (matching original code)
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        # frozen_epochs controls the transition; finetune duration is implied by epoch_count
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


irap_local_rec_trainer_multiscale = TrainerConfig(
    eval_step=MultiScaleSupervisedStep(scales=(1.0, 0.75, 1 / 0.75), amp=True),
    train_step=SupervisedStep(amp=True),  # Single-scale during training
    loss=MultiAttributeCrossEntropyLoss(),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


# Semi-supervised common kwargs
_semisup_kl_div = None


def _get_semisup_kl_div():
    global _semisup_kl_div
    if _semisup_kl_div is None:
        from vidlu_irap_gaim.training.semisup import multi_attribute_kl_div_ll

        _semisup_kl_div = multi_attribute_kl_div_ll
    return _semisup_kl_div


# We need the loss at module load time for TrainerConfig, so import it directly
from vidlu_irap_gaim.training.semisup import multi_attribute_kl_div_ll

irap_semisup_common_kwargs = dict(
    eval_step=SupervisedStep(eval=True, amp=True),
    train_step=SemisupConsStep(
        loss_cons=multi_attribute_kl_div_ll,
        amp=True,
        alpha=1.0,  # unsupervised loss weight
    ),
    loss=MultiAttributeCrossEntropyLoss(),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
)


irap_semisup_trainer = TrainerConfig(
    **irap_semisup_common_kwargs,
    extension_fs=[
        partial(SemisupVAT, attack_f=partial(ColorJitterAttack, preset=JITTER_STRONG)),
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


irap_semisup_trainer_ph20 = TrainerConfig(
    **irap_semisup_common_kwargs,
    extension_fs=[
        partial(
            SemisupVAT,
            attack_f=partial(
                ph20_attack,
                step_count=0,
                loss=multi_attribute_kl_div_ll,
                output_to_target=lambda x: x,
            ),
        ),
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)

irap_semisup_trainer_ph3 = TrainerConfig(
    **irap_semisup_common_kwargs,
    extension_fs=[
        partial(
            SemisupVAT,
            attack_f=partial(
                ph3_attack,
                step_count=0,
                loss=multi_attribute_kl_div_ll,
                output_to_target=lambda x: x,
            ),
        ),
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


# =============================================================================
# Pseudo-Label Self-Training Trainers
# =============================================================================

irap_pseudo_label_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=True),
    train_step=MultiAttributePseudoLabelStep(
        conf_thresh=0.0,  # override per experiment
        temperature=1.0,  # override per experiment
        alpha=1.0,
        amp=True,
    ),
    loss=MultiAttributeCrossEntropyLoss(),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        partial(SemisupVAT, attack_f=partial(ColorJitterAttack, preset=JITTER_STRONG)),
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


def _make_nofreeze_trainer(config):
    """Removes FreezeThenFinetune from a trainer config, replacing its per-phase
    optimizers and MultiplicativeLR decay with the config-level optimizer and a
    single cosine schedule over all epochs."""
    return TrainerConfig(
        config,
        extension_fs=[
            ext
            for ext in config["extension_fs"]
            if not (callable(ext) and getattr(ext, "func", None) is FreezeThenFinetune)
        ],
        lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    )


irap_local_rec_trainer_nofreeze = _make_nofreeze_trainer(irap_local_rec_trainer)
irap_semisup_trainer_ph3_nofreeze = _make_nofreeze_trainer(irap_semisup_trainer_ph3)
irap_semisup_trainer_ph20_nofreeze = _make_nofreeze_trainer(irap_semisup_trainer_ph20)
irap_pseudo_label_trainer_nofreeze = _make_nofreeze_trainer(irap_pseudo_label_trainer)


irap_pseudo_label_offline_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=True),
    train_step=SupervisedStep(amp=True),
    loss=MultiAttributeCrossEntropyLoss(ignore_index=-1),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        DynamicBalancedRecallWeights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)


# =============================================================================
# VLM Fine-tuning Trainer
# =============================================================================


# Import VLM-specific steps (lazy import to avoid loading heavy dependencies)
def _get_vlm_train_step():
    from vidlu_irap_gaim.vlm.finetuning.steps import VLMTrainStep

    return VLMTrainStep(amp=True, gradient_accumulation_steps=4)


def _get_vlm_eval_step():
    from vidlu_irap_gaim.vlm.finetuning.steps import VLMEvalStep

    return VLMEvalStep(amp=True)


def trainable_parameters_optimizer(optimizer_f):
    @wraps(optimizer_f)
    def wrapper(params, *args, **kwargs):
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "No trainable parameters found. "
                "Ensure the VLM classifier's initialize() was called before Trainer creation."
            )
        return optimizer_f(trainable, *args, **kwargs)

    return wrapper


# Transformer-backbone trainers ####################################################################
#
# One config per fine-tuning regime, all backbone-agnostic: the backbone comes from the model
# string and the country from the data string. None of them uses `FreezeThenFinetune` – the
# regime is fixed for the whole run, and `EncoderOptimizerMaker` sets the encoder's trainability
# and builds the parameter groups. They all derive from `irap_local_rec_trainer_nofreeze`, so
# each states only what distinguishes it.

# The batch of `irap_local_rec_trainer`, so that switching backbones changes one thing rather
# than two. A 3-frame example is 3 backbone images, so this is 36 images per step: roughly 4 GB
# for ViT-B and 12 GB for ViT-L under full fine-tuning, or ~6 GB with `grad_checkpointing=True`.
# Batch size and learning rate are coupled – MAE scales its rate as `blr * batch/256` while the
# rates below are absolute – so overriding this changes the effective recipe.
_vit_batch_size = 12


# Linear probing: the backbone is a fixed feature extractor and only the heads and a parametric
# pooling head (e.g. SigLIP's attention pooling) are trained. The cheapest comparison of raw
# feature quality between backbones, and for DINOv3 the evaluated regime rather than merely the
# cheap one (https://arxiv.org/abs/2508.10104). The LR is high because everything trained is
# randomly initialized. DINOv3's own linear evaluation sweeps it over 1e-5..1e-1
# (`dinov3/eval/linear.py`), so this single value is worth a sweep.
irap_vit_linear_probe = TrainerConfig(
    irap_local_rec_trainer_nofreeze,
    optimizer_f=EncoderOptimizerMaker(torch.optim.AdamW, trainable="pool",
                                      lr=1e-3, weight_decay=1e-4),
    batch_size=_vit_batch_size,
)


# Full fine-tuning with layer-wise LR decay: earlier blocks get exponentially smaller rates.
#
# `layer_decay` and `weight_decay` are MAE's fine-tuning values (https://arxiv.org/abs/2111.06377,
# FINETUNE.md): 0.65 and 0.05 for ViT-B, 0.75 and 0.05 for ViT-L/H. The decay compounds over
# depth, hence one config per depth: at 0.65 a 24-block ViT-L would train its first block at
# ~3e-5 of the base rate.
#
# The schedule must be multiplicative (see `WarmupCosineLR`): `CosineLR`'s `eta_min` is an
# absolute floor and would flatten the layer-wise decay.
_irap_vit_llrd = TrainerConfig(
    irap_local_rec_trainer_nofreeze,
    lr_scheduler_f=partial(WarmupCosineLR, warmup_proportion=0.1, min_factor=0.01),
    batch_size=_vit_batch_size,
)

# Config inheritance overrides a whole field, so what varies *inside* `optimizer_f` is bound
# here instead: the arguments the LLRD regimes share, with the differing one or two given at
# each use. That leaves every `EncoderOptimizerMaker` argument reachable.
_vit_llrd_optimizer_f = partial(EncoderOptimizerMaker, torch.optim.AdamW, trainable="all",
                                lr=1e-4, weight_decay=0.05,
                                head_lr_multiplier=10., head_weight_decay=1e-4)


# ViT-B backbones (SigLIP 2 / SPAR).
irap_vit_finetune_llrd = TrainerConfig(
    _irap_vit_llrd, optimizer_f=_vit_llrd_optimizer_f(layer_decay=0.65))
# ViT-L backbones (DINOv3-L): shallower decay per MAE. Build the encoder with
# `grad_checkpointing=True` – at this batch it takes ViT-L from ~12 GB to ~6 GB.
#
# These are MAE's ImageNet-1k values and DINOv3 publishes no full-fine-tuning recipe, but the
# transplant works: on Vietnam (10,818 examples) this reaches 0.4542 amF1 against 0.4334 for
# `irap_vit_partial_finetune` and 0.4189 for `irap_vit_linear_probe`, at a run-to-run spread
# of ~0.007. Prefer `epoch_count=5`: amF1 is flat to 20 epochs while the validation loss
# degrades badly. See the README for the full table.
irap_vitl_finetune_llrd = TrainerConfig(
    _irap_vit_llrd, optimizer_f=_vit_llrd_optimizer_f(layer_decay=0.75))


# Partial fine-tuning: only the last few transformer blocks, the output norm and the pooling
# head. The usual recommendation when the backbone is far larger than the dataset, though on
# iRAP-Vietnam it lands between the probe and the full fine-tune rather than above them (0.4334
# amF1 against 0.4189 and 0.4542). `num_trainable_blocks` is set on the encoder (default 2, as
# in SPAR's own fine-tuning configuration).
irap_vit_partial_finetune = TrainerConfig(
    _irap_vit_llrd,
    optimizer_f=_vit_llrd_optimizer_f(layer_decay=0.75, trainable="last_blocks"))


# LoRA: only the low-rank adapters injected into the backbone's transformer blocks are trained.
# Requires an encoder built with a LoRA rank (`lora_r=` for `TimmViTEncoder`, always for
# `Qwen3VLVisionEncoder`). The LR is an order of magnitude above a full fine-tune's, which
# adapters tolerate. `batch_size` is 4 because this config's original user is the
# 8-billion-parameter Qwen3-VL vision tower; a timm ViT-B/L under LoRA can use 12.
irap_vit_lora = TrainerConfig(
    irap_local_rec_trainer_nofreeze,
    optimizer_f=EncoderOptimizerMaker(torch.optim.AdamW, trainable="lora",
                                      lr=1e-4, weight_decay=1e-4),
    batch_size=4,
)


def without_dynamic_weights(config):
    """Drops `DynamicBalancedRecallWeights`, leaving the cross-entropy unweighted.

    For ablations isolating the effect of the recall weighting. (It is no longer needed
    under `combined_train_loader_f`: the extension now pools priors over every `train*`
    split.)
    """
    # `getattr(..., "func", ...)` sees through a `partial` of the class.
    return TrainerConfig(
        config,
        extension_fs=[ext for ext in config["extension_fs"]
                      if getattr(ext, "func", ext) is not DynamicBalancedRecallWeights],
    )


irap_vit_linear_probe_nodyn = without_dynamic_weights(irap_vit_linear_probe)
irap_vit_finetune_llrd_nodyn = without_dynamic_weights(irap_vit_finetune_llrd)
irap_vitl_finetune_llrd_nodyn = without_dynamic_weights(irap_vitl_finetune_llrd)
irap_vit_partial_finetune_nodyn = without_dynamic_weights(irap_vit_partial_finetune)
irap_vit_lora_nodyn = without_dynamic_weights(irap_vit_lora)


# VLM fine-tuning trainer configuration
# Uses loss as proxy metric during training; full generation eval done separately
vlm_finetune_trainer = TrainerConfig(
    train_step=_get_vlm_train_step(),
    eval_step=_get_vlm_eval_step(),
    # Loss is computed inside the train_step, not by Trainer
    loss=lambda out, target, reduction="mean": torch.tensor(0.0),
    optimizer_f=trainable_parameters_optimizer(
        partial(torch.optim.AdamW, lr=1e-5, weight_decay=0.1)
    ),
    epoch_count=10,
    eval_count=10,  # Evaluate every epoch
    batch_size=2,
    eval_batch_size=1,
    extension_fs=[
        MultiAttributeScorePrinter,
    ],
)


# Gemma-4-tuned VLM trainer.  Hyperparameter sources:
#
#   [unsloth-g4]   Unsloth, "Gemma 4 Fine-tuning Guide" (LoRA recipe)
#                  https://unsloth.ai/docs/models/gemma-4/train
#   [hf-vertex]    HF / Google Cloud, "Fine-tune Gemma 4 with TRL on Vertex AI"
#                  (full-FT TRL/SFTTrainer recipe for `gemma-4-E2B-it`)
#                  https://huggingface.co/docs/google-cloud/examples/vertex-ai-notebooks-fine-tune-gemma-4
#   [hf-carla]     huggingface/huggingface-gemma-recipes, `scripts/carla_vlm_gemma.py`
#                  (LoRA-on-multimodal Gemma 4 reference, default r=128 alpha=256)
#                  https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/carla_vlm_gemma.py
#   [unsloth-4907] unslothai/unsloth#4907 — Gemma 4 26B-A4B MoE LoRA "abnormally
#                  low trainable param count" bug (motivates including expert
#                  layers in LoRA targets).
#                  https://github.com/unslothai/unsloth/issues/4907
#   [empirical]    Measured on this repo's 4× A6000 BIH dataset run.
#
# Per-hyperparameter justification:
#
#   - ``lr=1e-4``: Unsloth's LoRA-on-Gemma-4 recipe uses 2e-4 [unsloth-g4];
#     HF-vertex and HF-CARLA use 5e-6 because they're full FT, not LoRA
#     [hf-vertex, hf-carla].  1e-4 is the conservative LoRA midpoint; LoRA
#     adapters tolerate (and need) a higher LR than full FT because their
#     parameter count is tiny relative to the base.
#   - ``weight_decay=1e-3``: Unsloth's recipe value for Gemma 4 [unsloth-g4].
#     [hf-vertex] does not set weight_decay (HF SFTConfig default ≈ 0).
#     The prior 1e-1 used by `vlm_finetune_trainer` is too aggressive for
#     LoRA — heavy WD pushes adapters back toward zero.
#   - ``fused=True``: PyTorch ≥2.0 AdamW fused kernel.  General best practice
#     on Ampere; harmless on stacks without fused support
#     (https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
#   - ``epoch_count=3``: matches [hf-vertex] (`num_train_epochs=3`).
#   - ``batch_size=2``: [empirical].  [hf-vertex] uses 4 per device on H100
#     80 GB; on 4× A6000 with bf16-naive-MP we have less per-shard headroom.
#     With ``gradient_accumulation_steps=4`` (in ``_get_vlm_train_step``,
#     matching [hf-carla]) the effective batch size is 8.
#   - ``eval_batch_size=1``: [empirical] — with bf16 base sharded across
#     the 4× A6000s via ``device_map="auto"`` (naive MP), the teacher-forced
#     eval forward at B=6 OOMs.  Generative metric eval is per-sample inside
#     ``_generate_and_parse_batch`` regardless of this value, so B=1 carries
#     no metric-quality cost.
#   - ``eval_count=3``: evaluate every epoch (vidlu convention, same as the
#     other trainers in this file).
#
# See also .devdocs/gemma4_26b_a4b_finetuning_resources.md for the full
# hardware/resource budget and the comparison table behind these picks.
gemma4_vlm_finetune_trainer = TrainerConfig(
    train_step=_get_vlm_train_step(),
    eval_step=_get_vlm_eval_step(),
    loss=lambda out, target, reduction="mean": torch.tensor(0.0),
    optimizer_f=trainable_parameters_optimizer(
        partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-3, fused=True)
    ),
    epoch_count=10,
    batch_size=2,
    eval_batch_size=1,
    # In-training eval = teacher-forced loss only (set VLM_SKIP_GENERATIVE_EVAL=1).
    # Generative metric scoring is deferred to scripts/eval_generative_gemma.py
    # so that the slow per-sample autoregressive decode does not dominate
    # training wall-clock.  See plan §"Phase 1 — Cheap wins".
    eval_count=10,
    extension_fs=[
        MultiAttributeScorePrinter,
    ],
)


# Public API: every TrainerConfig defined above, plus the data-loader factory and the
# helper for deriving variants of a config.
__all__ = ["combined_train_loader_f", "without_dynamic_weights"] + [
    name for name, value in globals().items() if isinstance(value, TrainerConfig)
]
