import dataclasses as dc
import typing as T
import copy
import os

import torch

from vidlu.training.steps import SemisupCleanTargetConsStepBase

from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from .jitter import make_sequence_color_jitter, JITTER_STRONG


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
        from vidlu_irap_gaim.training.semisup import get_hard_pseudo_labels, update_adaptive_thresholds
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
