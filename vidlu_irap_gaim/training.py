from functools import partial, wraps
import dataclasses as dc
import typing as T
import copy
import os

import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from torchvision.transforms import transforms as T_trans

from vidlu.configs.training import TrainerConfig
from vidlu_irap_gaim.dynamic import DynamicBalancedRecallWeights
from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from vidlu_irap_gaim.semisup import multi_attribute_kl_div_ll
from vidlu.training.steps import SupervisedStep, SemisupConsStep, SemisupCleanTargetConsStepBase
from vidlu.training.extensions import SemisupVAT
from vidlu_irap_gaim.visualization import VisualizationExtension
from vidlu_irap_gaim.extensions import MultiAttributeScorePrinter
from vidlu.training.extensions import TrainerExtension
from vidlu.configs.robustness import ph20_attack, ph3_attack


class FreezeThenFinetune(TrainerExtension):
    def __init__(
        self,
        num_frozen_epochs: int = 2,
        frozen_lr: float = 5e-5,
        finetune_lr: float = 1e-5,
        frozen_weight_decay: float = 1e-3,
        finetune_weight_decay: float = 1e-3,
        frozen_lr_scheduler_f: T.Callable = partial(MultiplicativeLR, lr_lambda=lambda epoch: 0.8),
        finetune_lr_scheduler_f: T.Callable = partial(MultiplicativeLR, lr_lambda=lambda epoch: 0.88),
    ):
        # Phase lengths (the trainer's epoch_count should normally match frozen+finetune)
        self.num_frozen_epochs = num_frozen_epochs

        # Optimizer / scheduler hyperparameters per phase
        self.frozen_lr = frozen_lr
        self.finetune_lr = finetune_lr
        self.frozen_wd = frozen_weight_decay
        self.finetune_wd = finetune_weight_decay
        self.frozen_lr_scheduler_f = frozen_lr_scheduler_f
        self.finetune_lr_scheduler_f = finetune_lr_scheduler_f

    def initialize(self, trainer):
        def reinit_optimizer_and_scheduler(lr, wd, lr_scheduler):
            # preserve optimizer state? start fresh as in original code
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=lr,
                weight_decay=wd,
            )
            trainer.optimizer = opt
            trainer.lr_scheduler = lr_scheduler(optimizer=opt)

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
                reinit_optimizer_and_scheduler(self.frozen_lr, self.frozen_wd, self.frozen_lr_scheduler_f)
            else:  # finetune
                # Enable all parameters in finetune phase
                for p in trainer.model.parameters():
                    p.requires_grad = True
                reinit_optimizer_and_scheduler(self.finetune_lr, self.finetune_wd, self.finetune_lr_scheduler_f)

        # Initialize in frozen phase at start
        set_phase("frozen")

        # Handle resumption: ensure phase is correct before loading optimizer state
        original_load_state_dict = trainer.load_state_dict

        def load_state_dict_wrapper(state_dict):
            # Check the epoch in the checkpoint
            # state_dict['training'] is the EpochLoop state dict, containing 'epoch'
            if "training" in state_dict and "epoch" in state_dict["training"]:
                epoch = state_dict["training"]["epoch"]
                # If we finished the frozen epochs, we should be in finetune phase
                # (e.g. if num_frozen_epochs=2, epochs 0 and 1 are frozen.
                #  If we save at end of epoch 1, stored epoch is 1. Next is 2.
                #  Transition happens at START of epoch 2.
                #  So if stored epoch >= 2, we are definitely in finetune.
                #  Wait, if stored epoch is 1, we are at end of frozen. Optimizer is frozen.
                #  If stored epoch is 2, we are at end of first finetune epoch. Optimizer is finetune.
                if epoch >= self.num_frozen_epochs:
                    set_phase("finetune")
                else:
                    set_phase("frozen")

            original_load_state_dict(state_dict)

        trainer.load_state_dict = load_state_dict_wrapper

        @trainer.training.epoch_started.handler
        def on_epoch_started(state):
            if state.epoch == self.num_frozen_epochs:
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
    from .attrs import get_attrs_to_include

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


@dc.dataclass
class MultiScaleSupervisedStep:
    """Eval step with multi-scale inference and probability averaging.

    Applies the model at multiple scales, converts logits to probabilities,
    and averages probabilities across scales for each attribute.
    """

    scales: T.Sequence[float] = (1.0, 0.75, 1 / 0.75)
    amp: bool = False

    def __call__(self, trainer, batch):
        from vidlu.training.steps import _unify_sup_batch, untag
        from vidlu_irap_gaim.models.multiscale import MultiScaleSequenceInference
        from vidlu.utils.collections import NameDict
        import contextlib as ctx

        model = trainer.model
        model.eval()

        # Lazy-create the multi-scale wrapper
        if not hasattr(self, "_ms_model") or self._ms_model is None:
            self._ms_model = MultiScaleSequenceInference(model, scales=self.scales)

        amp_ctx = torch.cuda.amp.autocast() if self.amp else ctx.nullcontext()
        with amp_ctx:
            with torch.no_grad():
                x, y = _unify_sup_batch(batch)[:2]
                probs = self._ms_model(untag(x))  # Tuple of averaged probabilities
                loss = trainer.loss(probs, y, reduction="mean")

        return NameDict(x=x, target=y, out=probs, loss=loss.item())


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


@dc.dataclass
class MultiAttributePseudoLabelStep(SemisupCleanTargetConsStepBase):
    """Pseudo-label self-training step for multi-attribute classification.

    Model output must be a tuple of (B, K_i) logit tensors; targets must be
    (B, A) integer tensors. Uses a frozen pre-trained teacher (loaded from
    checkpoint) to generate hard argmax pseudo-labels with per-(sample, attribute)
    confidence thresholding and temperature scaling.

    FixMatch-style: teacher runs on clean x_u, student is trained on jittered x_u
    (requires SemisupVAT(ColorJitterAttack) extension in the trainer config).
    """
    pre_trained_teacher: T.Optional[T.Union[str, os.PathLike, torch.nn.Module]] = None
    temperature: float = 1.0
    conf_thresh: T.Union[float, dict] = 0.0
    eval_mode_teacher: bool = True  # always True for frozen teacher
    _teacher: T.Optional[torch.nn.Module] = dc.field(default=None, repr=False, compare=False)

    def get_student_and_teacher(self, trainer):
        model = trainer.model
        if self._teacher is None:
            if isinstance(self.pre_trained_teacher, (str, os.PathLike)):
                path = self.pre_trained_teacher
                if isinstance(path, str) and path.startswith('$'):
                    path = os.environ[path[1:]]
                teacher = copy.deepcopy(model)
                params = torch.load(path, map_location='cpu')
                teacher.load_state_dict(params)
            elif self.pre_trained_teacher is None:
                teacher = model  # self-training: student is its own teacher
            else:
                teacher = self.pre_trained_teacher
            teacher.eval()
            teacher.requires_grad_(False)
            self._teacher = teacher
        if self._teacher is not model:
            # Ensure frozen teacher is on same device as model
            model_dev = next(model.parameters()).device
            if next(self._teacher.parameters()).device != model_dev:
                self._teacher.to(model_dev)
        return model, self._teacher

    def _get_cons_loss_and_output_to_target(self, attack):
        from vidlu_irap_gaim.semisup import get_hard_pseudo_labels, update_adaptive_thresholds
        loss_cons = MultiAttributeCrossEntropyLoss(ignore_index=-1)

        temperature = self.temperature
        conf_thresh = self.conf_thresh  # can be float or dict
        adaptive_thresholds = {}  # track per-attribute adaptive thresholds if conf_thresh is dict

        def output_to_target(out_u):
            nonlocal adaptive_thresholds
            # Use adaptive thresholds if conf_thresh is a dict; otherwise use fixed value
            thresh_to_use = adaptive_thresholds if isinstance(conf_thresh, dict) else conf_thresh
            labels, _ = get_hard_pseudo_labels(out_u, temperature=temperature,
                                               conf_thresh=thresh_to_use)
            # Update adaptive thresholds for next iteration if in adaptive mode
            if isinstance(conf_thresh, dict):
                adaptive_thresholds = update_adaptive_thresholds(
                    out_u, adaptive_thresholds, ema_momentum=0.999
                )
            return labels

        return loss_cons, output_to_target


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
