import copy
import dataclasses as dc
import os
import typing as T

import torch

from vidlu.training.steps import AmpMixin, SemisupCleanTargetConsStepBase
from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss

from .jitter import JITTER_STRONG, make_sequence_color_jitter


@dc.dataclass
class MultiScaleSupervisedStep(AmpMixin):
    """Supervised step with multi-scale inference and probability averaging.

    Applies the model at every scale in `scales`, converts each scale's logits to
    probabilities and averages them per attribute.

    `out` holds the averaged probabilities, so the metrics need `output_kind='probs'`, and
    `loss` is their negative log-likelihood. The loss is computed on the log of the averaged
    probabilities because `trainer.loss` (`MultiAttributeCrossEntropyLoss`) applies
    `log_softmax`, which is the identity on log-probabilities. Passing the probabilities
    themselves would take a softmax of a softmax and report a wrong number.

    `eval` selects the role, as in `vidlu.training.steps.SupervisedStep`: evaluation puts the
    model in eval mode and runs without gradients; training keeps gradients and takes an
    optimization step. Training backpropagates through one forward pass per scale, so it
    costs about `len(scales)` times the time and memory of a single-scale step.
    """

    scales: T.Sequence[float] = (1.0, 0.75, 1 / 0.75)
    eval: bool = False
    _ms_model: torch.nn.Module | None = dc.field(default=None, init=False, repr=False,
                                                 compare=False)

    def __call__(self, trainer, batch):
        import contextlib as ctx

        from vidlu.training.steps import _unify_sup_batch, untag
        from vidlu.utils.collections import NameDict
        from vidlu_irap_gaim.models.multiscale import MultiScaleSequenceInference

        model = trainer.model
        model.eval() if self.eval else model.train()

        if self._ms_model is None:
            self._ms_model = MultiScaleSequenceInference(model, scales=self.scales)

        with self.maybe_amp(), torch.no_grad() if self.eval else ctx.suppress():
            x, y = _unify_sup_batch(batch)[:2]
            probs = self._ms_model(untag(x))  # Tuple of averaged probabilities
            # `clamp_min` keeps `log` finite where a scale gives a class zero probability.
            log_probs = tuple(p.clamp_min(torch.finfo(p.dtype).tiny).log() for p in probs)
            loss = trainer.loss(log_probs, y, reduction="mean")

        if not self.eval:
            self.do_optimization_step(trainer.optimizer, loss)

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
    pre_trained_teacher: T.Union[str, os.PathLike, torch.nn.Module] | None = None
    temperature: float = 1.0
    conf_thresh: float | dict = 0.0
    eval_mode_teacher: bool = True  # always True for frozen teacher
    _teacher: torch.nn.Module | None = dc.field(default=None, repr=False, compare=False)

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
