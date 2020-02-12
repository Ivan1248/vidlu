import dataclasses as dc
from dataclasses import dataclass
from functools import partial
from numbers import Number
import warnings
from time import sleep
import typing as T

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from vidlu.modules.losses import NLLLossWithLogits, KLDivLossWithLogits
from vidlu import ops
from vidlu.utils.misc import Event
import vidlu.modules.inputwise as vmi
import vidlu.modules as vm
import vidlu.optim as vo
import vidlu.utils.torch as vut


# Prediction transformations #######################################################################

def _to_hard_target(output):
    """Creates discrete labels from model output with arg-max along the first
    dimension.

    Args:
        output: model output (logits).

    Returns:
        A tensor containing predicted labels.
    """
    _, y = torch.max(output, dim=1)
    return y


def _to_soft_target(output, temperature=1):
    """Computes softmax output given logits `output`.

     It computes `softmax(output/temperature)`. `temperature` is `1` by default.
     Lower temperature gives harder labels.

    Args:
        output: model output (logits).
        temperature (float): softmax temperature.

    Returns:
        A tensor containing predicted probabilities.
    """
    return F.softmax(output if temperature == 1 else output / temperature, dim=1)


# Early stopping conditions ########################################################################

class PredictionSimilarityIndicator:
    def __call__(self, state) -> torch.BoolTensor:
        raise NotImplemented()


@dataclass
class ClassMatches(PredictionSimilarityIndicator):
    def __call__(self, state) -> torch.BoolTensor:
        return state.output.argmax(dim=1) == state.y


@dataclass
class LossIsBelowThreshold(PredictionSimilarityIndicator):
    threshold: float
    loss: callable = None

    def __call__(self, state) -> torch.BoolTensor:
        loss = state.loss if self.loss is None else self.loss(state.output, state.y)
        return loss < self.threshold


# AttackState ######################################################################################

@dataclass
class AttackState:
    x: torch.Tensor
    y: torch.Tensor
    x_adv: torch.Tensor
    y_adv: torch.Tensor
    output: torch.Tensor
    loss: torch.Tensor
    loss_sum: float
    reg_loss_sum: float
    grad: torch.Tensor
    step: int = None
    loss_mean: float = dc.field(init=False)
    reg_loss_mean: float = dc.field(init=False)
    pert_model: torch.nn.Module = None

    def __post_init__(self):
        self.loss_mean = self.loss_sum / len(self.x)
        self.reg_loss_mean = self.reg_loss_sum / len(self.x)


# get_prediction ###################################################################################

_to_target_wiki = dict(
    soft=_to_soft_target,
    hard=_to_hard_target)


# Attack ###########################################################################################


@torch.no_grad()
def _perturbation_model_from_perturbation(pert_or_x_adv, x=None):
    pert = pert_or_x_adv - x if x is not None else pert_or_x_adv
    pert_model = vmi.Additive(())
    pert_model(pert)
    pert_model.addend.set_(pert)
    return pert_model


@dataclass
class Attack:
    """Adversarial attack (input attack) base class.

    Args:
        loss ((Tensor, Tensor) -> Tensor) a loss function that returns losses
        without reduction (sum or mean).
        minimize (bool): a value telling whther the loss sould be miniumized.
            Default: `False`
        clip_bounds (tuple): a pair representing the minimum and maximum values
            (scalars or arrays) that the input has to be constrained to.
        to_virtual_target (Tensor -> Tensor): a function that¸turns the output
            of the model into a prediction that can be used as a target label in
            the second argument in the loss. Default: a function that returns a
            (hard) classifier prediction -- argmax over dimension 1 of the
            output.

    Note:
        `clip_bounds` should be set to `None` (default) if adversarial inputs
         do not have to be clipped.
    """
    loss: T.Callable = dc.field(default_factory=partial(NLLLossWithLogits, ignore_index=-1))
    minimize: bool = False
    clip_bounds: tuple = (0, 1)
    to_virtual_target: T.Union[T.Callable, str] = _to_hard_target

    on_virtual_label_computed = Event()
    compute_model_grads: bool = False

    def __post_init__(self):
        if isinstance(self.to_virtual_target, str):
            self.to_virtual_target = _to_target_wiki[self.to_virtual_target]

    def __call__(self, model, x, y=None, output=None, **kwargs):
        """Generates an adversarial perturbation model.

        `attack(model, x, y)(x)` should give the same result as
        `attack.perturb(model, x, y)`.

        Args:
            model (Module): model.
            x (Tensor): input tensor.
            y (Tensor, optional): target/label tensor. If `None`, the prediction
                obtained from `self.to_virtual_target(output)` is used as the
                label.
            output (Tensor, optional): output for use as a target. If `None`
                it is obtained by calling `model(x)` (and then transformad with
                `self.to_virtual_target`).

        Return:
            (torch.nn.Module) An adversarial perturbation model with paameters
        """

        if y is None:
            y = self._get_virtual_target(model, output, x)
        if type(self)._get_perturbation is not Attack._get_perturbation:
            pert = self._get_perturbation(model, x, y=y, **kwargs)
            return (_perturbation_model_from_perturbation(pert) if isinstance(pert, torch.Tensor)
                    else pert)
        elif type(self)._perturb is not Attack._perturb:
            x_adv = self._perturb(model, x, y=y, **kwargs)
            return _perturbation_model_from_perturbation(x_adv, x)

    def perturb(self, model, x, y=None, output=None, **kwargs):
        """Generates an adversarial example.

        Args:
            model (Module): model.
            x (Tensor): input tensor.
            y (Tensor, optional): target/label tensor. If `None`, the prediction
                obtained from `self.to_virtual_target(output)` is used as the
                label.
            output (Tensor, optional): output for use as a target. If `None`
                it is obtained by calling `model(x)` (and then transformad with
                `self.to_virtual_target`).

        Return:
            Perturbed input.
        """
        if y is None:
            y = self._get_virtual_target(model, output, x)
        if type(self)._perturb is not Attack._perturb:
            return self._perturb(model, x, y=y, **kwargs)
        elif type(self)._get_perturbation is not Attack._get_perturbation:
            return self._get_perturbation(model, x, y=y, **kwargs)(x)

    def _get_virtual_target(self, model, output, x):
        with torch.no_grad():
            output = model(x) if output is None else output
            y = self.to_virtual_target(output)
            self.on_virtual_label_computed(dict(output=output, y=y))
            return y

    def _get_output_and_loss_s_and_grad(self, model, x, y=None):
        x.requires_grad_()
        output = model(x)
        loss = self.loss(output, y).view(len(x), -1).mean(1).sum()
        loss.backward()
        return output, loss, x.grad

    def _clip(self, x):
        return x.clamp(*self.clip_bounds)

    def _perturb(self, model, x, y=None, **kwargs):
        # has to be implemented in subclasses
        raise NotImplementedError()

    def _get_perturbation(self, model, x, y=None, output=None, **kwargs):
        # has to be implemented in subclasses
        raise NotImplementedError()


@dataclass
class EarlyStoppingMixin:
    stop_on_success: bool = False
    similar: T.Callable = dc.field(default_factory=ClassMatches)


class DummyAttack(Attack):
    def _perturb(self, model, x, y=None, backward_callback=None):
        output, loss_s, grad = self._get_output_and_loss_s_and_grad(model, x, y)
        if backward_callback is not None:
            backward_callback(
                AttackState(x=x, y=y, output=output, x_adv=x, loss_sum=loss_s.item(), grad=grad,
                            loss=None, reg_loss_sum=0))
        return x


# GradientSignAttack ###############################################################################

@dataclass
class GradientSignAttack(Attack):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572
    """
    eps: float = 8 / 255

    def _perturb(self, model, x, y=None):
        output, _, grad = self._get_output_and_loss_s_and_grad(model, x, y)
        with torch.no_grad:
            return self._clip(x + self.eps * grad.sign())


@torch.no_grad()
def rand_init_delta(x, p, eps, bounds, batch=False):
    """Generates a random perturbation from a unit p-ball scaled by eps.

    Args:
        x (Tensor): input.
        p (Real): p-ball p.
        eps (Real or Tensor):
        bounds (Real or Tensor): clipping bounds.
        batch: whether `x` is a batch.

    """
    kw = dict(dtype=x.dtype, device=x.device)
    if batch:  # TODO: optimize
        delta = torch.stack(tuple(ops.random.uniform_sample_from_p_ball(p, x.shape[1:], **kw)
                                  for _ in range(x.shape[0])))
    else:
        delta = ops.random.uniform_sample_from_p_ball(p, x.shape, **kw)
    delta = delta.mul_(eps)

    return x + delta if bounds is None else torch.clamp(x + delta, *bounds) - x


# Attack step updates ##############################################################################


class AttackStepUpdate:
    def __call__(self, delta, grad):
        """Updates `delta` and returns the same object (modified)."""
        raise NotImplementedError()

    def init(self, delta, x):
        return delta


class DummyUpdate(AttackStepUpdate):
    def __call__(self, delta, grad):
        return delta


# Attacks loop templates and updates (steps) #######################################################


@torch.no_grad()
def zero_grad(delta):
    if delta.grad is not None:
        delta.grad.zero_()


def perturb_iterative(model, x, y, step_count, update, loss, minimize=False,
                      initial_pert=None, clip_bounds=(0, 1), similar=None, backward_callback=None):
    """Iteratively optimizes the loss over the input.

    Args:
        model: Forward pass function.
        x: Inputs.
        y: Input labels.
        step_count: Number of iterations.
        update (procedure): A `AttackStepUpdate` object defining a single
            perturbation update in an iteration, and optionally a finish
            function applied to the perturbation after the last optimization
            step.
        loss: a loss function.
        minimize (bool): Whether the attack should minimize the loss.
        initial_pert (optional): Initial perturbation.
        clip_bounds (optional): Minimum and maximum input value pair.
        similar (optional): a function that tells whether the attack is
            successful example-wise based on the predictions and the true
            labels. If it is provided, optimization is stopped when `similar`
            returns `False`. Otherwise,`step_count` of iterations is performed
            on every example.
        backward_callback: A callback function called after the gradient has
        been computed. It can be used for e.g. updating model parameters during
        input optimization (adversarial training for free).

    Returns:
        Perturbed inputs.
    """
    stop_on_success = similar is not None
    loss_fn, rloss_fn = loss if isinstance(loss, T.Sequence) else (loss, None)
    backward_callback = backward_callback or (lambda _: None)

    delta = torch.zeros_like(x) if initial_pert is None else initial_pert.detach().clone()
    delta.requires_grad_()

    if stop_on_success:  # store arrays because they shrink as attacks succeed
        x_all, delta_all, index = x, delta.detach().clone(), torch.arange(len(x))

    for i in range(step_count):
        x_adv = x + delta
        output = model(x_adv)
        loss = loss_fn(output, y).view(len(x), -1).mean(1).sum()
        reg_loss = rloss_fn(x, delta, output, y).view(len(x), -1).mean(1).sum() if rloss_fn \
            else torch.tensor(0.)
        zero_grad(delta)
        ((-loss if minimize else loss) - reg_loss).backward()  # maximized

        state = AttackState(x=x, y=y, output=output, x_adv=x_adv, loss_sum=loss.item(),
                            reg_loss_sum=reg_loss.item(), grad=delta.grad, step=i)
        backward_callback(state)

        with torch.no_grad():
            if stop_on_success:
                is_adv = similar(state) if minimize else ~similar(state)
                if is_adv.any():  # keep the already successful adversarial examples unchanged
                    delta_all[index[is_adv]] = delta[is_adv]
                    is_nonadv = ~is_adv
                    x, y, index, grad, delta = (a[is_nonadv] for a in
                                                [x, y, index, delta.grad, delta])
                    del is_nonadv  # free some memory
                    delta.requires_grad_()
                    delta.grad = grad
                    if is_adv.all():
                        break  # stop when all adversarial examples are successful
                del is_adv  # free some memory

            del state, loss, reg_loss, output  # free some memory
            delta = update(delta, delta.grad)

            if clip_bounds is not None:
                delta.add_(x).clamp_(*clip_bounds).sub_(x)

    with torch.no_grad():
        if stop_on_success:
            delta_all[index] = delta
            x, delta = x_all, delta_all

    return delta


def _reduce_loss(loss):
    return loss.view(loss.shape[0], -1).mean(1).sum()


def _reduce_losses(unred_loss, unred_reg_loss=None, nonadv_mask=None):
    """Used by `perturb_iterative_with_perturbation_model` to compute (masked)
    losses."""
    loss_nomask = _reduce_loss(unred_loss)
    loss = _reduce_loss(unred_loss * nonadv_mask) if nonadv_mask is not None else loss_nomask
    reg_loss = (torch.tensor(0., device=loss.device) if unred_reg_loss is None else
                loss if nonadv_mask is None else
                _reduce_loss(unred_reg_loss * nonadv_mask))
    return loss, reg_loss, loss_nomask


def _set_tensor_storage(tensor, storage, grad=None):
    tensor.set_(storage)
    tensor.grad = grad
    return tensor


def _extract_nonadv(pmodel, adv_mask, masking_mode, pm_params_all, index, y, x, prune):
    """Used by `perturb_iterative_with_perturbation_model` to optionally prune
    batches and mask gradients."""
    mask_loss, mask_grad = 'loss' in masking_mode and not prune, 'grad' in masking_mode

    if adv_mask.any():  # keep the already successful adversarial examples unchanged
        mask_loss &= len(adv_mask.shape) > 1
        is_adv = adv_mask if len(adv_mask.shape) == 1 else adv_mask.view(len(x), -1).all(dim=1)

        if prune and is_adv.any():
            # extract non-adversarial inputs and perturbations
            isnot_adv, index_adv = ~is_adv, index[is_adv]
            for p_all, p in zip(pm_params_all, pmodel.parameters()):
                p_all[index_adv] = p[is_adv]
                _set_tensor_storage(p, p[isnot_adv], grad=p.grad[isnot_adv])

            for a in (x, y, index):
                a.set_(a[isnot_adv])

            if (mask_loss or mask_grad) and len(adv_mask.shape) > 1:
                adv_mask.set_(adv_mask[isnot_adv])

        if mask_grad and len(adv_mask.shape) > 1:
            for p in pmodel.parameters():
                p.grad[adv_mask] = 0
    else:  # nothing adversarial, no need for masking
        mask_loss = False
    return adv_mask.logical_not_() if mask_loss else None


def perturb_iterative_with_perturbation_model_hacky(
        model, x, y, step_count, optim_f, loss_fn, minimize=False, pert_model=None,
        projection=None, clip_bounds=(0, 1), similar=None, masking_mode='loss', prune=True,
        backward_callback=None):
    """Iteratively optimizes the loss over the input.
    Compared to `perturb_iterative`, uses an optimizer instead of an update
    function and stopping on success (when `similar` is provided) does not
    speed up optimization unless the optimization succeeds on all inputs.

    Args:
        model: Forward pass function.
        x: Inputs.
        y: Input labels.
        step_count: (Max.) number of iterations.
        optim_f: optimizer.
        loss: A loss function that returns one ore more values per input.
        minimize (bool): Whether the attack should minimize the loss.`
        pert_model: A function returning a perturbation model that has "batch
            parameters" of type `vidlu.BatchParameter`. If `None`, an
            elementwise perturbation model is created.
        clip_bounds (optional): Minimum and maximum input value pair.
        similar (optional): A function that tells whether the attack is
            successful example-wise based on the predictions and the true
            labels. If it is provided, optimization is stopped when `similar`
            returns `False`. Otherwise,`step_count` of iterations is performed
            on every example. Besides the batch size, the shape of the output
            can also be the shape of the loss function output (if `masking_mode`
            is `'loss'`) or the input shape (if `masking_mode` is `'grad'`).
        masking_mode (Literal['loss', 'grad']): If `similar` is provided and
            returns multiple values per input, `masking_mode` determines whether
            loss or the gradient is to be masked for early stopping of the
            optimization.
        prune (bool): if True and the `similar` function is provided, batches
            are pruned when possible to speed up computation.
        backward_callback: A callback function called after the gradient has
            been computed. It can be used for e.g. updating model parameters
            during input optimization (adversarial training for free).

    Returns:
        The perturbation model.
    """
    if prune:
        warnings.warn("If pruning is enabled, this procedure relies on an hack used to avoid a race"
                      " condition without guarantee. Set `prune=False` if you want to guarantee no"
                      " errors but make it slower.")
    stop_on_success = similar is not None
    loss_fn, reg_loss_fn = loss_fn if isinstance(loss_fn, T.Sequence) else (loss_fn, None)
    backward_callback = backward_callback or (lambda _: None)

    pmodel = pert_model or vmi.Additive(())
    pmodel(x)  # initialoze perturbation model (parameter shapes have to be inferred from x)
    pmodel.train()
    optim = optim_f(pmodel.parameters())
    projection = projection or (lambda _: None)

    if stop_on_success:
        nonadv_mask = None
        if prune:
            with torch.no_grad():
                x_all = x
                index, x, y = torch.arange(len(x)), x.detach().clone(), y.detach().clone()
                pm_params_all = [p.detach().clone() for p in pmodel.parameters()]
        else:
            pm_params_all, index = None, None
    else:
        masking_mode, pm_params_all = '', None

    for i in range(step_count):
        x_adv = pmodel(x)
        output = model(x_adv)

        loss, reg_loss, loss_no_mask = _reduce_losses(
            unred_loss=loss_fn(output, y),
            unred_reg_loss=reg_loss_fn(pmodel, x, y, x_adv, output) if reg_loss_fn else None,
            nonadv_mask=nonadv_mask)
        optim.zero_grad()
        ((loss if minimize else -loss) + reg_loss).backward()  # minimized

        state = AttackState(x=x, y=y, output=output, x_adv=x_adv, loss_sum=loss_no_mask.item(),
                            reg_loss_sum=reg_loss.item(), grad=None, step=i, pert_model=pmodel)
        backward_callback(state)
        del output, loss, reg_loss, loss_no_mask

        with torch.no_grad():
            if stop_on_success:
                nonadv_mask = _extract_nonadv(
                    pmodel, similar(state) if minimize else ~similar(state),
                    masking_mode, pm_params_all, index, y, x, prune=prune)
            del state  # free some memory and remove references to the graph

            optim.step()
            projection(pmodel)

            if clip_bounds is not None and torch.any(x_adv.clamp_(*clip_bounds) != x_adv):
                warnings.warn(f"The perturbed input has values out of bounds {clip_bounds}.")
            if len(x) == 0:
                break
            if i < step_count - 1:
                del x_adv  # request graph deletion by removing the last reference
                sleep(0.002)  # a hack to give PyTorch time to reset Autograd metadata

    if stop_on_success and prune:  # store the final parameters of the unsuccesful examples
        with torch.no_grad():
            x = x_all
            for p_all, p in zip(pm_params_all, pmodel.parameters()):
                p_all[index] = p
                p.set_(p_all)

    pmodel.eval()
    with torch.no_grad():
        return pmodel, pmodel(x)


def _get_mask_and_update_index(pmodel, adv_mask, masking_mode, index):
    """Used by `perturb_iterative_with_perturbation_model` to optionally prune
    batches and mask gradients."""
    mask_loss, mask_grad = 'loss' in masking_mode, 'grad' in masking_mode

    if adv_mask.any():  # keep the already successful adversarial examples unchanged
        if len(adv_mask.shape) == 1:
            isnot_adv = adv_mask.logical_not_()
            mask_loss = False
        else:
            isnot_adv = adv_mask.view(len(adv_mask), -1).all(dim=1).logical_not_()
            if mask_loss or mask_grad:
                adv_mask.set_(adv_mask[isnot_adv])
                if mask_grad:
                    for p in pmodel.parameters():
                        p.grad[index][adv_mask] = 0
        index.set_(index[isnot_adv])
    else:
        mask_loss = None
    return adv_mask.logical_not_() if mask_loss else None


def perturb_iterative_with_perturbation_model(
        model, x, y, step_count, optim_f, loss_fn, minimize=False, pert_model=None,
        projection=None, clip_bounds=(0, 1), similar=None, masking_mode='loss',
        backward_callback=None, compute_model_grads=False):
    """Iteratively optimizes the loss over the input.
    Compared to `perturb_iterative`, uses an optimizer instead of an update
    function and stopping on success (when `similar` is provided) does not
    speed up optimization unless the optimization succeeds on all inputs.

    Args:
        model: Forward pass function.
        x: Inputs.
        y: Input labels.
        step_count: (Max.) number of iterations.
        optim_f: optimizer factory. Optimizers with running stats are not
            supported.
        loss_fn: A loss function that returns one ore more values per input.
            minimize (bool): Whether the attack should minimize the loss.`
        pert_model: A function returning a perturbation model that has "batch
            parameters" of type `vidlu.BatchParameter`. If `None`, an
            elementwise perturbation model is created.
        clip_bounds (optional): Minimum and maximum input value pair.
        similar (optional): A function that tells whether the attack is
            successful example-wise based on the predictions and the true
            labels. If it is provided, optimization is stopped when `similar`
            returns `False`. Otherwise,`step_count` of iterations is performed
            on every example. Besides the batch size, the shape of the output
            can also be the shape of the loss function output (if `masking_mode`
            is `'loss'`) or the input shape (if `masking_mode` is `'grad'`).
        masking_mode (Literal['loss', 'grad']): If `similar` is provided and
            returns multiple values per input, `masking_mode` determines whether
            loss or the gradient is to be masked for early stopping of the
            optimization.
        backward_callback: A callback function called after the gradient has
            been computed. It can be used for e.g. updating model parameters
            during input optimization (adversarial training for free).

    Returns:
        The perturbation model.
    """

    stop_on_success = similar is not None
    loss_fn, reg_loss_fn = loss_fn if isinstance(loss_fn, T.Sequence) else (loss_fn, None)
    backward_callback = backward_callback or (lambda _: None)

    pmodel = pert_model or vmi.Additive(())
    pmodel(x)  # initialize perturbation model (parameter shapes have to be inferred from x)
    pmodel.train()

    optim = optim_f(pmodel.parameters())  # optimizers with running stats are not supported
    projection = projection or (lambda _: None)

    if stop_on_success:
        with torch.no_grad():
            index = torch.arange(len(x))
    else:
        masking_mode = ''
    nonadv_mask = None

    for i in range(step_count):
        x_adv, y_ = pmodel(x, y)

        if stop_on_success:
            storage = [x, y, x_adv]
            x, x_adv = [a[index] for a in (x, x_adv)]
            y, y_ = (y[index],) * 2 if y_ is y else (a[index] for a in (y, y_))

        with vut.switch_requires_grad(model, compute_model_grads):
            output = model(x_adv)
            unred_loss = loss_fn(output, y_)
            loss, reg_loss, loss_no_mask = _reduce_losses(
                unred_loss=unred_loss,
                unred_reg_loss=reg_loss_fn(pmodel, x, y_, x_adv, output) if reg_loss_fn else None,
                nonadv_mask=nonadv_mask)
            optim.zero_grad()
            ((loss if minimize else -loss) + reg_loss).backward()  # minimized

        # print(i, pmodel.gamma.grad.abs().mean(), pmodel.gamma.unique())

        state = AttackState(
            x=x, y=y, output=output, x_adv=x_adv, y_adv=y_, loss=unred_loss,
            loss_sum=loss_no_mask.item(), reg_loss_sum=reg_loss.item(), grad=None, step=i,
            pert_model=pmodel)
        backward_callback(state)
        del output, loss, reg_loss, loss_no_mask  # free some memory

        with torch.no_grad():
            if stop_on_success:
                nonadv_mask = _get_mask_and_update_index(
                    pmodel, similar(state) if minimize else ~similar(state), masking_mode, index)
            del state  # free some memory

            optim.step()
            projection(pmodel)

            if clip_bounds is not None:
                x_adv.clamp_(*clip_bounds)

        if stop_on_success:
            x, y, x_adv = storage
            if len(index) == 0:
                break

    pmodel.eval()
    return pmodel


# Attacks ##########################################################################################

@dataclass
class PGDUpdate(AttackStepUpdate):
    """PGD step update
    Args:
        step_size: attack step size per iteration.
        eps: maximum distortion.
        grad_processing (Union[Callable, str] =): preprocessing of gradient
            before multiplication with step_size.'sign' for gradient sign,
            'normalize' for p-norm normalization.
        project_or_p (Union[Number, type(np.inf), Callable]): a function that
            constrains the perturbation to the threat model or a number
            determining the p ot the p-ball to project to..
    """
    step_size: float
    eps: float
    grad_processing: T.Union[T.Callable, str] = torch.sign
    project_or_p: dc.InitVar[T.Union[Number, type(np.inf), T.Callable]] = np.inf
    # derived
    project: T.Callable = dc.field(init=False)

    def __post_init__(self, project_or_p):
        self.project = (partial(ops.batch.project_to_p_ball, p=project_or_p)
                        if isinstance(project_or_p, (Number, type(np.inf)))
                        else project_or_p)
        if isinstance(self.grad_processing, str):
            self.grad_processing = vo.get_grad_processing(self.grad_processing)

    def __call__(self, delta, grad):
        pgrad = self.grad_processing(grad)
        delta += pgrad.mul_(self.step_size)
        return delta.set_(self.project(delta, self.eps))  # TODO: check for p in {1, 2}


@dataclass
class PGDAttackOld(EarlyStoppingMixin, Attack):
    """The PGD attack (Madry et al., 2017).

    The attack performs `step_count` steps of size `step_size`, while always 
    staying within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    See the documentation of perturb_iterative.
    """
    eps: float = 8 / 255
    step_size: float = 2 / 255
    step_count: int = 40
    grad_processing: str = 'sign'
    rand_init: bool = True
    p: float = np.inf

    def _perturb(self, model, x, y=None, initial_pert=None, backward_callback=None):
        if initial_pert is None:
            initial_pert = (rand_init_delta(x, self.p, self.eps, self.clip_bounds) if self.rand_init
                            else torch.zeros_like(x))

        update = PGDUpdate(step_size=self.step_size, eps=self.eps,
                           grad_processing=self.grad_processing, project_or_p=self.p)
        delta = perturb_iterative(
            model, x, y, step_count=self.step_count, update=update, loss=self.loss,
            minimize=self.minimize, initial_pert=initial_pert, clip_bounds=self.clip_bounds,
            similar=self.similar if self.stop_on_success else None,
            backward_callback=backward_callback)

        return self._clip(x + delta)


@dataclass
class PerturbationModelAttack(EarlyStoppingMixin, Attack):
    pert_model_f: vmi.PerturbationModel = partial(vmi.Additive, ())
    optim_f: T.Callable = partial(vo.ProcessedGradientDescent, process_grad=torch.sign)
    initializer: T.Callable[[vmi.PerturbationModel], None] = None
    projection: T.Union[float, T.Callable[[vmi.PerturbationModel], None]] = 0.1
    step_size: T.Union[float, T.Mapping[str, float]] = 0.05
    step_count: int = 40

    def __post_init__(self):
        if not callable(self.projection):
            eps = self.projection

            def projection(pmodel):
                params = pmodel.named_parameters()
                default_vals = vmi.named_default_parameters(pmodel, full_size=False)
                for (name, p), (name_, d) in zip(params, default_vals):
                    assert name == name_  # TODO: remove
                    p.clamp_(min=d - eps, max=d + eps)

            self.projection = projection

    def _get_perturbation(self, model, x, y=None, pert_model=None, initialize_pert_model=True,
                          backward_callback=None):
        with torch.no_grad():
            if pert_model is None:
                pert_model = self.pert_model_f()
                pert_model(x)
            if initialize_pert_model and self.initializer is not None:
                self.initializer(pert_model)

        if isinstance(self.step_size, T.Mapping):
            optim_f = lambda p: self.optim_f(
                [dict(params=[vm.get_submodule(pert_model, name)], lr=step)
                 for name, step in self.step_size.items()])
        else:
            optim_f = partial(self.optim_f, lr=self.step_size)

        return perturb_iterative_with_perturbation_model(
            model, x, y, step_count=self.step_count, optim_f=optim_f, loss_fn=self.loss,
            minimize=self.minimize, pert_model=pert_model, projection=self.projection,
            clip_bounds=self.clip_bounds, similar=self.similar if self.stop_on_success else None,
            backward_callback=backward_callback, compute_model_grads=self.compute_model_grads)


@dataclass
class PGDAttack(EarlyStoppingMixin, Attack):
    """The PGD attack (Madry et al., 2017).

    The attack performs `step_count` steps of size `step_size`, while always
    staying within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    See the documentation of perturb_iterative.
    """
    eps: float = 8 / 255
    step_size: float = 2 / 255
    step_count: int = 40
    rand_init: bool = True
    p: float = np.inf
    optim_f = partial(vo.ProcessedGradientDescent, process_grad=torch.sign)

    @torch.no_grad()
    def _project_params(self, pmodel):
        pmodel.addend.set_(ops.batch.project_to_p_ball(pmodel.addend, r=self.eps, p=self.p))

    @torch.no_grad()
    def _pert_model_init(self, pmodel):
        if self.rand_init:
            pmodel.addend.set_((rand_init_delta(pmodel.addend, self.p, self.eps, self.clip_bounds)))
        else:
            pmodel.addend.zero_()

    def __post_init__(self):
        fields = (f.name for f in dc.fields(PerturbationModelAttack) if f.init)
        self._base = PerturbationModelAttack(
            **{k: getattr(self, k) for k in fields if hasattr(self, k)},
            pert_model_f=partial(vmi.Additive, ()), pert_model_init=self._pert_model_init,
            projection=self._project_params)

    def _get_perturbation(self, model, x, y=None, initial_pert=None, backward_callback=None):
        p = partial(self._base._get_perturbation, model, x, y, backward_callback=backward_callback)
        if initial_pert is not None:
            pert_model = vmi.Additive(())
            pert_model.addend.data = initial_pert
            return p(pert_model)
        return p()


@dataclass
class PGDLDSAttack(PGDAttack):
    to_virtual_target: T.Callable = _to_soft_target
    loss: T.Callable = dc.field(default_factory=KLDivLossWithLogits)


@dataclass
class VATUpdate(AttackStepUpdate):
    eps: float  # final perturbation size
    xi: float  # optimization perturation size
    p: Number

    def __call__(self, delta, grad):
        return delta.set_(ops.batch.normalize_by_norm(grad, self.p).mul_(self.xi))


@dataclass
class VATAttack(Attack):
    to_virtual_target: T.Callable = _to_soft_target
    loss: T.Callable = dc.field(default_factory=KLDivLossWithLogits)

    # TODO: set default CIFAR eps, xi and initial_pert
    eps: float = 10  # * 448/32
    xi: float = 1e-6  # * 448/32
    step_count: int = 1
    p: Number = 2

    def _perturb(self, model, x, y=None, backward_callback=None):
        initial_pert = rand_init_delta(x, self.p, self.xi, self.clip_bounds)
        update = VATUpdate(xi=self.xi, eps=self.eps, p=self.p)
        delta = perturb_iterative(model, x, y, step_count=self.step_count, update=update,
                                  loss=self.loss, minimize=self.minimize, initial_pert=initial_pert,
                                  clip_bounds=self.clip_bounds, backward_callback=backward_callback)
        delta = ops.batch.normalize_by_norm(delta, self.p, inplace=True).mul_(self.eps)
        return x + delta


# DDN ##############################################################################################


@dataclass
class DDN(EarlyStoppingMixin, Attack):
    """DDN attack: decoupling the direction and norm of the perturbation to
    achieve a small L2 norm in few steps.

    Args:
        eps_init (float, optional): Initial value for the norm.
        step_size (float): Optimization step size.
        max_step_count (int): Maxmimum number of steps for the optimization.
        gamma (float, optional): Factor by which the norm will be modified.
            new_norm = norm * (1 + or - gamma).
        p (float): p-norm p. If specified,
    """
    eps_init: float = 1.
    step_size: float = 2 / 255
    max_step_count: int = 200
    gamma: float = 0.1
    grad_processing: str = 'sign'
    rand_init: bool = True
    p: float = 2

    def _perturb(self, model, x, y=None, backward_callback=None):
        n = x.shape[0]
        delta = torch.zeros_like(x, requires_grad=True)
        norm = torch.full((n,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(x, 1 - x).view(n, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_step_count,
                                                         eta_min=0.01)

        best_norm = worst_norm.clone()
        best_delta = torch.zeros_like(x)
        adv_found = torch.zeros(x.size(0), dtype=torch.uint8, device=x.device)

        for i in range(self.max_step_count):
            scheduler.step()

            curr_norm = delta.data.view(n, -1).norm(p=self.p, dim=1)
            x_adv = x + delta
            output = model(x_adv)
            loss = self.loss(output, y).view(len(x), -1).mean(1).sum()
            grad = delta.grad  # .detach().clone() unnecessary

            state = AttackState(x=x, y=y, output=output, x_adv=x_adv, loss_sum=loss.item(),
                                grad=grad, step=i)
            backward_callback(state)

            is_adv = self.similar(state) if self.minimize else ~self.similar(state)
            is_smaller = curr_norm < best_norm
            is_smaller_adv = is_adv ^ is_smaller
            adv_found[is_smaller_adv] = True
            best_norm[is_smaller_adv] = curr_norm[is_smaller_adv]
            best_delta[is_smaller_adv] = delta.data[is_smaller_adv]

            optimizer.zero_grad()
            loss.backward()

            # renorming gradient
            grad_norms = ops.batch.norm(grad, p=self.p)
            delta.grad.div_(ops.batch.redim_as(grad_norms, grad))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(n, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)

        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return x + best_delta


# CW ###############################################################################################
# TODO

'''
CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10


def get_carlini_loss(targeted, confidence_threshold):
    def carlini_loss(logits, y, l2distsq, c):
        y_onehot = ops.one_hot(y, logits.shape[-1])
        real = (y_onehot * logits).sum(dim=-1)

        other = ((1.0 - y_onehot) * logits - (y_onehot * TARGET_MULT)).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if targeted:
            loss1 = (other - real + confidence_threshold).relu_()
        else:
            loss1 = (real - other + confidence_threshold).relu_()
        loss2 = l2distsq.sum()
        loss1 = torch.sum(c * loss1)
        loss = loss1 + loss2
        return loss

    return carlini_loss


class CarliniWagnerL2Attack(Attack):
    def __init__(self, loss=None, clip_bounds=None, similar=None,
                 get_prediction=_to_hard_target, distance_fn=ops.batch.l2_distace_sqr,
                 num_classes=None, confidence=0,
                 learning_rate=0.01, binary_search_steps=9, max_iter=10000, abort_early=True,
                 initial_const=1e-3):
        """The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644

        Args:
            num_classes: number of clasess.
            confidence: confidence of the adversarial examples.
            learning_rate: the learning rate for the attack algorithm
            binary_search_steps: number of binary search times to find the optimum
            max_iter: the maximum number of iterations
            abort_early: if set to true, abort early if getting stuck in local min
            initial_const: initial value of the constant c
        """
        if loss is not None:
            raise NotImplementedError("The CW attack currently does not support a different loss"
                                      " function other than the default. Setting loss manually"
                                      " is not effective.")
        loss = loss or get_carlini_loss()
        super().__init__(loss, clip_bounds, similar, get_prediction)

        self.distance_fn = distance_fn
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP

    def _loss(self, output, y_onehot, l2distsq, loss_coef):
        return get_carlini_loss(self.targeted, self.confidence)(output, y_onehot, l2distsq,
                                                                loss_coef)

    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        # label

        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)), label] -= self.confidence
            else:
                output[torch.arange(len(label)), label] += self.confidence
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()

        return self.similar(pred, label)

    def _forward_and_update_delta(self, model, optimizer, x_atanh, delta, y_onehot, loss_coeffs):
        optimizer.zero_grad()

        adv = ops.scaled_tanh(delta + x_atanh, *self.clip_bounds)
        l2distsq = self.distance_fn(adv, ops.scaled_tanh(x_atanh, *self.clip_bounds))
        output = model(adv)

        loss = self._loss(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.detach(), output.detach(), adv.detach()

    def _arctanh_clip(self, x):
        result = ops.clamp((x - self.clip_min) / (self.clip_max - self.clip_min), min=self.clip_min,
                           max=self.clip_max) * 2 - 1
        return ops.atanh(result * ONE_MINUS_EPS)

    def _update_if_smaller_dist_succeed(self, adv_img, labs, output, l2distsq, batch_size,
                                        cur_l2distsqs, cur_labels, final_l2distsqs, final_labels,
                                        final_advs):
        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (l2distsq < cur_l2distsqs) & self._is_successful(output_logits, target_label, True)

        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (l2distsq < final_l2distsqs) & self._is_successful(output_logits, target_label, True)
        final_l2distsqs[mask] = l2distsq[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]

    def _update_loss_coeffs(self, labs, cur_labels, batch_size, loss_coeffs, coeff_upper_bound,
                            coeff_lower_bound):
        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def _perturb(self, model, x, y):
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = torch.full_like(coeff_lower_bound, CARLINI_COEFF_UPPER)
        loss_coeffs = torch.full_like(y, self.initial_const, dtype=torch.float)
        final_advs = x
        x_atanh = self._arctanh_clip(x)
        y_onehot = ops.one_hot(y, self.num_classes).float()

        final_l2distsqs = torch.full((batch_size,), CARLINI_L2DIST_UPPER, device=x.device)
        final_labels = torch.full((batch_size,), INVALID_LABEL, dtype=torch.int, device=x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = torch.full_like(final_l2distsqs, CARLINI_L2DIST_UPPER)
            cur_labels = torch.full_like(final_labels, INVALID_LABEL)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iter):
                loss, l2distsq, output, adv_img = self._forward_and_update_delta(
                    model, optimizer, x_atanh, delta, y_onehot, loss_coeffs)
                if self.abort_early and ii % (self.max_iter // NUM_CHECKS or 1) == 0:
                    if loss > prevloss * ONE_MINUS_EPS:
                        break
                    prevloss = loss

                self._update_if_smaller_dist_succeed(adv_img, y, output, l2distsq, batch_size,
                                                     cur_l2distsqs, cur_labels, final_l2distsqs,
                                                     final_labels, final_advs)

            self._update_loss_coeffs(y, cur_labels, batch_size, loss_coeffs, coeff_upper_bound,
                                     coeff_lower_bound)

        return final_advs
'''
