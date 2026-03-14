from functools import partial, wraps

import torch

from vidlu.configs.training import TrainerConfig
from vidlu.training.steps import SupervisedStep, SemisupConsStep
from vidlu.training.extensions import SemisupVAT
from vidlu.configs.robustness import ph20_attack, ph3_attack

from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from .dynamic_weights import DynamicBalancedRecallWeights
from .extensions import FreezeThenFinetune, MultiAttributeScorePrinter, VisualizationExtension
from .jitter import make_sequence_color_jitter, JITTER_STRONG
from .steps import MultiScaleSupervisedStep, MultiAttributePseudoLabelStep, ColorJitterAttack


def _make_dynamic_balanced_recall_weights(dirs):
    """Factory function for DynamicBalancedRecallWeights.

    Args:
        dirs: Experiment directories object (required). This is automatically provided
            from the experiment via the factory namespace. The cache_dir is extracted
            from dirs.cache (handling the case where it might be a list).
    """
    # Handle case where dirs.cache might be a list (use first element)
    cache_dir = dirs.cache[0] if isinstance(dirs.cache, (list, tuple)) else dirs.cache
    from vidlu_irap_gaim.data.attrs import get_attrs_to_include

    attrs_to_include = get_attrs_to_include()
    return DynamicBalancedRecallWeights(cache_dir=cache_dir, attrs_to_include=attrs_to_include)


# Basic classification trainer with supervised step
# Loss is supplied externally as multi-attribute wrapper (see factory usage)
# Default configuration matches train_local_rec_paper.sh: 2 frozen + 8 finetune = 10 total epochs
epoch_count = 2 + 8  # 2 frozen + 8 finetune (matches paper script)
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
        conf_thresh=0.0,    # override per experiment
        temperature=1.0,    # override per experiment
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
                "Ensure Qwen3VLClassifier.initialize() was called before Trainer creation."
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
    optimizer_f=trainable_parameters_optimizer(partial(torch.optim.AdamW, lr=1e-5, weight_decay=0.1)),
    epoch_count=3,
    batch_size=2,
    eval_batch_size=6,
    eval_count=3,  # Evaluate every epoch
    extension_fs=[
        MultiAttributeScorePrinter,
    ],
)
