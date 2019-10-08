from collections import Callable
import dataclasses as dc
from dataclasses import dataclass
from functools import partial, wraps
from numbers import Number
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from vidlu.modules.loss import SoftmaxCrossEntropyLoss
from vidlu import ops
from vidlu.utils.misc import Event


def _predict_hard(output):
    """Compute predicted labels given `x`. Used to prevent label leaking

    Args:
        predict: a model.
        x (Tensor): an input for the model.

    Returns:
        A tensor containing predicted labels.
    """
    _, y = torch.max(output, dim=1)
    return y


def _predict_soft(output, temperature=1):
    """Computes softmax output given logits `x`.

     It computes `softmax(x/temperature)`. `temperature` is `1` by default.
     Lower temperature gives harder labels.

    Args:
        predict: a model.
        x (Tensor): an input for the model.
        temperature (float): softmax temperature.

    Returns:
        A tensor containing predicted labels.
    """
    return F.softmax(output if temperature == 1 else output / temperature, dim=1)


# PredictionMatchIndicator

class PredictionMatchIndicator:
    def __call__(self, state) -> torch.Tensor:  # -> BoolTensor
        raise NotImplemented()


@dataclass
class ClassMatches(PredictionMatchIndicator):
    def __call__(self, state) -> torch.Tensor:
        return state.output.argmax(dim=1) == state.y


@dataclass
class LossIsBelowThreshold(PredictionMatchIndicator):
    threshold: float
    loss: callable = None

    def __call__(self, state) -> torch.Tensor:
        loss = state.loss if self.loss is None else self.loss(state.output, state.y)
        return loss < self.threshold


# AttackState

@dataclass
class AttackState:
    x: torch.Tensor
    y: torch.Tensor
    x_adv: torch.Tensor
    output: torch.Tensor
    loss: torch.Tensor
    grad: torch.Tensor
    step: int = None


# get_prediction

_get_prediction_wiki = dict(
    soft=_predict_soft,
    hard=_predict_hard)


# Attack

@dataclass
class Attack:
    """Adversarial attack (input attack) base class.

    Args:
        model: a model.
        loss: a loss function that is increased by perturbing the input. It
            should be the negative loss if the attack is to be targeted.
            Reduction should be 'mean' because the loss might be used for
            updating model parameters or on dense prediction tasks.

        get_prediction (Tensor -> Tensor): a function that¸turns the output of
            the model into the predicted label that can be used as the second
            argument in the loss. Default: a function that returns a (hard)
            classifier prediction -- arggmax over dimension 1 of the output.

    Note:
        If `minimize_loss=True`, the base class constructor wraps `loss` so
        that the loss is multiplied by -1 so that subclasses thus don't have to
        check whether they need to maximize or minimize loss. When calling the
        base constructor, `clip_interval` should be set to `None` (default) if
        the output of `_predict` doesn't have to be clipped.
    """
    model: nn.Module
    loss: Callable = dc.field(default_factory=partial(SoftmaxCrossEntropyLoss, ignore_index=-1))
    minimize: bool = False
    clip_bounds: tuple = None
    get_prediction: Callable = _predict_hard
    matches: Callable = dc.field(default_factory=ClassMatches)

    on_virtual_output_computed = Event()

    def __post_init__(self):
        if type(self._perturb) == Attack._perturb:
            raise RuntimeError("The `_perturb` (not `perturb`) method should be overridden in"
                               + "f subclass `{type(self).__name__}`.")
        if hasattr(self.loss, 'reduction') and 'mean' not in self.loss.reduction:
            raise ValueError("Loss should be averaged over the batch.")
        if isinstance(self.get_prediction, str):
            self.get_prediction = _get_prediction_wiki[self.get_prediction]

    def perturb(self, x, y=None, **kwargs):
        """Generates an adversarial example.

        Args:
            x (Tensor): input tensor.
            y (Tensor): label tensor. If `None`, the prediction obtained from
                `get_prediction(model(x))` is used as the label.

        Return:
            Perturbed input.
        """
        if y is None:
            with torch.no_grad():
                output = self.model(x)
                y = self.get_prediction(output)
                self.on_virtual_output_computed(dict(output=output, y=y))
        x_adv = self._perturb(x.detach(), y.detach(), **kwargs)
        if self.clip_bounds is not None:
            x_adv = self._clip(x_adv)

        return x_adv

    def _get_output_loss_grad(self, x, y):
        x.requires_grad_()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        return output, loss, x.grad

    def _clip(self, x):
        return torch.clamp(x, *self.clip_bounds)

    def _perturb(self, x, y, **kwargs):
        # has to be implemented in subclasses
        raise NotImplementedError()


class DummyAttack(Attack):
    def _perturb(self, x, y=None, backward_callback=None):
        output, loss, grad = self._get_output_loss_grad(x, y)
        if backward_callback is not None:
            backward_callback(AttackState(x=x, y=y, output=output, x_adv=x, loss=loss,
                                          grad=grad))
        return x


# GradientSignAttack ###############################################################################

@dataclass
class GradientSignAttack(Attack):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572
    """
    eps: float = 8 / 255

    def _perturb(self, x, y=None):
        output, loss, grad = self._get_output_loss_grad(x, y)
        with torch.no_grad:
            return x + self.eps * grad.sign()


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


# Iterative ########################################################################################

def _get_grad_preprocessing(name):
    if name == 'sign':
        return lambda g: g.sign()
    elif name.startswith('normalize'):
        if '_' in name:
            p = name.split('_', 1)[1]
            p = np.inf if p == 'inf' else eval(p)
        else:
            p = 2
        return lambda g: g / ops.batch.norm(g, p, keep_dims=True)
    elif 'raw':  # multiplies ift by batch size because the loss is averaged
        return lambda g: g * len(g)


class AttackStepUpdate:
    def __call__(self, delta, grad, x):
        raise NotImplementedError()

    def finish(self, delta, x):
        return delta


class DummyUpdate(AttackStepUpdate):
    def __call__(self, delta, grad, x):
        return delta


@dataclass
class PGDUpdate(AttackStepUpdate):
    """PGD step update
    Args:
        step_size: attack step size per iteration.
        eps: maximum distortion.
        grad_preprocessing (Union[Callable, str] =): preprocessing of gradient
            before multiplication with step_size.'sign' for gradient sign,
            'normalize' for p-norm normalization.
        project_or_p (Union[Number, type(np.inf), Callable]): a function that
            constrains the perturbation to the threat model or a number
            determining the p ot the p-ball to project to..
    """
    step_size: float
    eps: float
    grad_preprocessing: Union[Callable, str] = torch.sign
    project_or_p: dc.InitVar[Union[Number, type(np.inf), Callable]] = np.inf

    project: Callable = dc.field(init=False)

    def __post_init__(self, project_or_p):
        self.project = (partial(ops.batch.project_to_p_ball, p=project_or_p)
                        if isinstance(project_or_p, (Number, type(np.inf)))
                        else project_or_p)
        if isinstance(self.grad_preprocessing, str):
            self.grad_preprocessing = _get_grad_preprocessing(self.grad_preprocessing)

    def __call__(self, delta, grad, x):
        pgrad = self.grad_preprocessing(grad)
        delta += pgrad.mul_(self.step_size)
        delta.set_(self.project(delta, self.eps))  # TODO: check for p in {1, 2}
        return delta


@dataclass
class VATUpdate(AttackStepUpdate):
    xi: float  # optimization perturation size
    eps: float  # final perturbation size
    p: Number

    def __call__(self, delta, grad, x):
        return delta.set_(ops.batch.normalize_by_norm(grad, self.p).mul_(self.xi))

    def finish(self, delta, x):
        return delta.set_(ops.batch.normalize_by_norm(delta, self.p).mul_(self.eps))


def perturb_iterative(x, y, model, step_count, update, loss, minimize=False, delta_init=None,
                      clip_bounds=(0, 1), matches=None, backward_callback=None):
    """Iteratively maximizes the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    Args:
        x: Inputs.
        y: Input labels.
        model: Forward pass function.
        step_count: Number of iterations.
        update (procedure): A `AttackStepUpdate` object defining a single
            perturbation update in an iteration, and optionally a finish
            function applied to the perturbation after thelast optimization
            step.
        loss: loss function.
        minimize (bool): Whether the attack should miniimize the loss.
        delta_init (optional): Initial delta.
        clip_bounds (optional): Mininum and maximum input value pair.
        matches (optional): a function that determines whether the attack is
            successful example-wise based on the predictions and the true
            labels. If it None, step_count of iterations is performed on every
            example.
        backward_callback: TODO

    Returns:
        Perturbed inputs.
    """
    stop_on_success = matches is not None
    loss_fn = loss
    backward_callback = backward_callback or (lambda x: None)

    delta = torch.zeros_like(x) if delta_init is None else delta_init.detach().clone()
    delta.requires_grad_()

    if stop_on_success:
        x_all, y_all, delta_all, index = x, y, delta.detach().clone(), torch.arange(len(x))

    for step in range(step_count):
        x_adv = x + delta
        output = model(x_adv)
        loss = loss_fn(output, y)
        loss.backward()
        grad = delta.grad.detach().clone()
        delta.grad.zero_()

        state = AttackState(x=x, y=y, output=output, x_adv=x, loss=loss, grad=grad, step=step)
        backward_callback(state)

        with torch.no_grad():
            if stop_on_success:
                success_mask = matches(state) if minimize else ~matches(state)
                if success_mask.any():  # keep the already succesful adversarial examples unchanged
                    fail_mask = ~success_mask
                    delta_all[index[success_mask]] = delta[success_mask]
                    x, y, index, grad = (a[fail_mask] for a in [x, y, index, grad])
                    delta = delta[fail_mask].detach().clone().requires_grad_()
                    if success_mask.all():
                        break  # stop when all adversarial examples are successful

            delta = update(delta, grad.neg_() if minimize else grad, x=x)
            if clip_bounds is not None:
                delta.set_((x + delta).clamp_(*clip_bounds).sub_(x))

    if stop_on_success:
        delta_all[index] = delta
        x, delta = x_all, delta_all

    x_adv = x + delta  # clipping already done
    return x_adv


@dataclass
class PGDAttack(Attack):
    """The PGD attack (Madry et al., 2017).

    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    See the documentation of perturb_iterative.
    """
    eps: float = 8 / 255
    step_size: float = 2 / 255
    step_count: int = 40
    grad_preprocessing: str = 'sign'
    rand_init: bool = True
    p: float = np.inf
    stop_on_success: bool = False

    def _perturb(self, x, y=None, backward_callback=None):
        delta_init = (rand_init_delta(x, self.p, self.eps, self.clip_bounds) if self.rand_init
                      else torch.zeros_like(x))
        update = PGDUpdate(step_size=self.step_size, eps=self.eps,
                           grad_preprocessing=self.grad_preprocessing, project_or_p=self.p)

        return perturb_iterative(
            x, y, self.model, step_count=self.step_count, update=update, loss=self.loss,
            minimize=self.minimize, clip_bounds=self.clip_bounds, delta_init=delta_init,
            matches=self.matches if self.stop_on_success else None,
            backward_callback=backward_callback)


@dataclass
class VATAttack(Attack):
    """The PGD attack (Madry et al., 2017).

    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    See the documentation of perturb_iterative.
    """
    eps: float = 8 / 255
    xi: float = 2 / 255
    step_count: int = 1
    p: Number = 2

    def _perturb(self, x, y=None, backward_callback=None):
        delta_init = (rand_init_delta(x, self.p, self.eps, self.clip_bounds) if self.rand_init
                      else torch.zeros_like(x))
        update = VATUpdate(xi=self.xi, eps=self.eps, p=self.p)

        return perturb_iterative(
            x, y, self.model, step_count=self.step_count, update=update, loss=self.loss,
            minimize=self.minimize, clip_bounds=self.clip_bounds, delta_init=delta_init,
            matches=self.matches if self.stop_on_success else None,
            backward_callback=backward_callback)


# CW ###############################################################################################

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
        loss2 = (l2distsq).sum()
        loss1 = torch.sum(c * loss1)
        loss = loss1 + loss2
        return loss

    return carlini_loss


class CarliniWagnerL2Attack(Attack):
    def __init__(self, model, loss=None, clip_bounds=None, matches=None,
                 get_prediction=_predict_hard, distance_fn=ops.batch.l2_distace_sqr,
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
        super().__init__(model, loss, clip_bounds, matches, get_prediction)

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

        return self.matches(pred, label)

    def _forward_and_update_delta(self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):
        optimizer.zero_grad()

        adv = ops.scaled_tanh(delta + x_atanh, *self.clip_bounds)
        l2distsq = self.distance_fn(adv, ops.scaled_tanh(x_atanh, *self.clip_bounds))
        output = self.model(adv)

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

    def _perturb(self, x, y):
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
                    optimizer, x_atanh, delta, y_onehot, loss_coeffs)
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


# DDN ##############################################################################################

@dataclass
class DDNAttack(Attack):
    eps_init: float = 8 / 255
    step_size: float = 2 / 255
    max_step_count: int = 40
    gamma: float = 0.1
    grad_preprocessing: str = 'sign'
    rand_init: bool = True
    p: float = np.inf
