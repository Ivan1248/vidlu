from functools import partial, wraps

import torch

from vidlu.configs.training import TrainerConfig
from vidlu.optim.lr_schedulers import CosineLR
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


def _make_dynamic_balanced_recall_weights(dirs):
    """Factory function for DynamicBalancedRecallWeights.

    Args:
        dirs: Experiment directories object (required). This is automatically provided
            from the experiment via the factory namespace. The cache_dir is extracted
            from dirs.cache (handling the case where it might be a list).
    """
    # Handle case where dirs.cache might be a list (use first element)
    cache_dir = dirs.cache[0] if isinstance(dirs.cache, (list, tuple)) else dirs.cache
    from irap_data.attrs import get_attrs_to_include

    attrs_to_include = get_attrs_to_include()
    return DynamicBalancedRecallWeights(
        cache_dir=cache_dir, attrs_to_include=attrs_to_include
    )


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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
        _make_dynamic_balanced_recall_weights,
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
    eval_batch_size=6,
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


# Public API: every TrainerConfig defined above, plus the data-loader factory.
__all__ = ["combined_train_loader_f"] + [
    name for name, value in globals().items() if isinstance(value, TrainerConfig)
]
