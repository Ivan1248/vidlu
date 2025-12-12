from functools import partial
import dataclasses as dc
import typing as T

import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from torchvision.transforms import transforms as T_trans

from vidlu.configs.training import TrainerConfig
from vidlu_irap_gaim.dynamic import DynamicBalancedRecallWeights
from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from vidlu_irap_gaim.semisup import multi_attribute_kl_div_ll
from vidlu.training.steps import SupervisedStep
from vidlu.training.steps import SemisupConsStep
from vidlu.training.extensions import SemisupVAT
from vidlu_irap_gaim.visualization import VisualizationExtension
from vidlu.training.extensions import TrainerExtension
from vidlu.configs.robustness import ph20_attack


class FreezeThenFinetune(TrainerExtension):
    def __init__(
        self,
        frozen_epochs: int = 2,
        frozen_lr: float = 5e-5,
        finetune_lr: float = 1e-5,
        frozen_weight_decay: float = 1e-3,
        finetune_weight_decay: float = 1e-3,
        frozen_scheduler: float = 0.8,
        finetune_scheduler: float = 0.88,
    ):
        # Phase lengths (the trainer's epoch_count should normally match frozen+finetune)
        self.frozen_epochs = frozen_epochs

        # Optimizer / scheduler hyperparameters per phase
        self.frozen_lr = frozen_lr
        self.finetune_lr = finetune_lr
        self.frozen_wd = frozen_weight_decay
        self.finetune_wd = finetune_weight_decay
        self.frozen_sched = frozen_scheduler
        self.finetune_sched = finetune_scheduler

    def initialize(self, trainer):
        def reinit_optimizer_and_scheduler(lr, wd, scheduler_gamma):
            # preserve optimizer state? start fresh as in original code
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=lr,
                weight_decay=wd,
            )
            trainer.optimizer = opt
            trainer.lr_scheduler = MultiplicativeLR(opt, lr_lambda=lambda ep: scheduler_gamma)

        def get_trainable_parameters():
            """Get trainable parameters following original logic: use model's method if available"""
            if hasattr(trainer.model, "get_trainable_parameters"):
                return trainer.model.get_trainable_parameters()
            else:
                # Fallback: heads + SPP + attention for frozen phase
                trainable_params = []
                heads = getattr(trainer.model, "heads", [])
                for head in heads:
                    trainable_params.extend(head.parameters())
                attn_blocks = getattr(trainer.model, "attn_blocks", None)
                if attn_blocks is not None:
                    trainable_params.extend(attn_blocks)
                spp = getattr(trainer.model, "spp", None)
                if spp is not None:
                    trainable_params.extend(spp.parameters())
                return trainable_params

        def set_phase(phase: str):
            # Freeze/unfreeze parameters following original logic
            if phase == "frozen":
                # First freeze all parameters
                for p in trainer.model.parameters():
                    p.requires_grad = False
                # Then enable only the trainable subset (heads + SPP)
                trainable_params = get_trainable_parameters()
                for p in trainable_params:
                    p.requires_grad = True
                reinit_optimizer_and_scheduler(self.frozen_lr, self.frozen_wd, self.frozen_sched)
            else:  # finetune
                # Enable all parameters in finetune phase
                for p in trainer.model.parameters():
                    p.requires_grad = True
                reinit_optimizer_and_scheduler(self.finetune_lr, self.finetune_wd, self.finetune_sched)

        # Initialize in frozen phase at start
        set_phase("frozen")

        @trainer.training.epoch_started.handler
        def on_epoch_started(state):
            if state.epoch == self.frozen_epochs:
                set_phase("finetune")


# Jitter presets (single source of truth)
JITTER_STANDARD = dict(brightness=0.6, contrast=0.3, saturation=0.2, hue=0.02)
JITTER_STRONG = dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)


def make_sequence_color_jitter(
    *,
    brightness: float = None,
    contrast: float = None,
    saturation: float = None,
    hue: float = None,
    preset: dict = None,
):
    """
    Color jitter for sequence tensors stored in records.

    Expects rgb frames in [0, 1] (normalization handled by input adapter).
    Applies ColorJitter frame-by-frame.

    Args:
        brightness, contrast, saturation, hue: Individual parameters (override preset).
        preset: A dict with jitter parameters (e.g. JITTER_STANDARD, JITTER_STRONG).
                If None, uses JITTER_STANDARD.
    """
    if preset is None:
        preset = JITTER_STANDARD

    # Allow individual params to override preset
    params = dict(preset)
    if brightness is not None:
        params["brightness"] = brightness
    if contrast is not None:
        params["contrast"] = contrast
    if saturation is not None:
        params["saturation"] = saturation
    if hue is not None:
        params["hue"] = hue

    color_jitter = T_trans.ColorJitter(**params)

    def _apply(record):
        if "rgb" not in record.keys():
            return record

        rgb = record["rgb"]
        if rgb.ndim not in (4, 5):
            return record

        if rgb.shape[0] == 0:
            return record

        if rgb.ndim == 4:
            jittered_frames = [color_jitter(frame) for frame in rgb]
            jittered = torch.stack(jittered_frames, dim=0)
        else:  # ndim == 5: (B, T, C, H, W)
            jittered_batches = []
            for b_idx in range(rgb.shape[0]):
                video = rgb[b_idx]
                jittered_frames = [color_jitter(frame) for frame in video]
                jittered_batches.append(torch.stack(jittered_frames, dim=0))
            jittered = torch.stack(jittered_batches, dim=0)

        return type(record)(record, rgb=jittered)

    return _apply


def _make_dynamic_balanced_recall_weights(dirs):
    """Factory function for DynamicBalancedRecallWeights.

    Args:
        dirs: Experiment directories object (required). This is automatically provided
            from the experiment via the factory namespace. The cache_dir is extracted
            from dirs.cache (handling the case where it might be a list).
    """
    # Handle case where dirs.cache might be a list (use first element)
    cache_dir = dirs.cache[0] if isinstance(dirs.cache, (list, tuple)) else dirs.cache
    return DynamicBalancedRecallWeights(cache_dir=cache_dir)


# Basic classification trainer with supervised step
# Loss is supplied externally as multi-attribute wrapper (see factory usage)
# Default configuration matches train_local_rec_paper.sh: 2 frozen + 8 finetune = 10 total epochs
epoch_count = 2 + 8  # 2 frozen + 8 finetune (matches paper script)
irap_local_rec_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=True),
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
        partial(FreezeThenFinetune, frozen_epochs=2),
        _make_dynamic_balanced_recall_weights,
        VisualizationExtension,
    ],
)


@dc.dataclass
class ColorJitterAttack:
    """Applies random photometric perturbations (Color Jitter).

    Acts as an "attack" (perturbation generator) for semi-supervised consistency.

    Args:
        preset: A dict with jitter parameters (e.g. JITTER_STANDARD, JITTER_STRONG).
                If None, uses JITTER_STRONG (stronger perturbations for semi-supervised).
    """

    preset: dict = None
    output_to_target: T.Callable = lambda x: x  # Identity mapping effectively
    loss: T.Callable = lambda *args: torch.tensor(0.0)  # Dummy loss

    def __post_init__(self):
        # Default to JITTER_STRONG for semi-supervised learning
        if self.preset is None:
            self.preset = JITTER_STRONG
        self.jitter = make_sequence_color_jitter(preset=self.preset)

    def __call__(self, model: torch.nn.Module, x, y=None, loss_mask=None, output=None, **kwargs):
        """Returns a perturbation model function."""
        # This needs to return a callable that takes (x, target, loss_mask)
        # and returns (x_p, target_p, loss_mask_p)
        # BUT based on SemisupConsStep usage:
        # perturb_x_u = lambda attack_target, loss_mask: _perturb_a(...)
        # _perturb_a calls: pmodel = attack(model, x, attack_target, loss_mask=loss_mask)
        # then x_p, target_p, loss_mask_p = pmodel(x, attack_target, loss_mask)

        # So we return self.perturb as the "pmodel" (which is a bit weird but fits the signature if we make it callable)
        return self.perturb

    def perturb(self, x, target=None, loss_mask=None):
        """Applies jitter to x.

        Note: Returns 2 values (x_p, target_p) when loss_mask is None,
        otherwise returns 3 values (x_p, target_p, loss_mask_p).
        This matches the interface expected by `_perturb_a` in steps.py.
        """
        # x is (B, T, C, H, W)
        if x.ndim != 5:  # Expecting (B, T, C, H, W)
            if loss_mask is None:
                return x, target
            return x, target, loss_mask

        # Wrap x in a dict to reuse make_sequence_color_jitter logic
        record = {"rgb": x}
        result_record = self.jitter(record)
        x_p = result_record["rgb"]

        if loss_mask is None:
            return x_p, target
        return x_p, target, loss_mask


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
        partial(FreezeThenFinetune, frozen_epochs=2),
        _make_dynamic_balanced_recall_weights,
        VisualizationExtension,
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
        partial(FreezeThenFinetune, frozen_epochs=2),
        _make_dynamic_balanced_recall_weights,
        VisualizationExtension,
    ],
)
