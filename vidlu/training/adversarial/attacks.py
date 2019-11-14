from collections import Callable
import dataclasses as dc
from dataclasses import dataclass
from functools import partial
from numbers import Number
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from vidlu.modules.losses import NLLLossWithLogits, KLDivLossWithLogits
from vidlu import ops
from vidlu.utils.misc import Event


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
    output: torch.Tensor
    loss_sum: float
    grad: torch.Tensor
    step: int = None
    loss_mean: float = dc.field(init=False)

    def __post_init__(self):
        self.loss_mean = self.loss_sum / len(self.x)


# get_prediction ###################################################################################

_to_target_wiki = dict(
    soft=_to_soft_target,
    hard=_to_hard_target)


# Attack ###########################################################################################

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
    loss: Callable = dc.field(default_factory=partial(NLLLossWithLogits, ignore_index=-1))
    minimize: bool = False
    clip_bounds: tuple = (0, 1)
    to_virtual_target: Union[Callable, str] = _to_hard_target

    on_virtual_label_computed = Event()

    def __post_init__(self):
        if self._perturb is Attack._perturb:
            raise RuntimeError("The `_perturb` (not `perturb`) method should be overridden in"
                               + "f subclass `{type(self).__name__}`.")
        if isinstance(self.to_virtual_target, str):
            self.to_virtual_target = _to_target_wiki[self.to_virtual_target]

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
            with torch.no_grad():
                output = model(x) if output is None else output
                y = self.to_virtual_target(output)
                self.on_virtual_label_computed(dict(output=output, y=y))
        x_adv = self._perturb(model, x.detach(), y.detach(), **kwargs)
        if self.clip_bounds is not None:
            x_adv = self._clip(x_adv)

        return x_adv

    def _get_output_and_loss_s_and_grad(self, model, x, y=None):
        x.requires_grad_()
        output = model(x)
        loss = self.loss(output, y).view(len(x), -1).mean(1).sum()
        loss.backward()
        return output, loss, x.grad

    def _clip(self, x):
        return torch.clamp(x, *self.clip_bounds)

    def _perturb(self, model, x, y=None, **kwargs):
        # has to be implemented in subclasses
        raise NotImplementedError()


@dataclass
class EarlyStoppingMixin:
    stop_on_success: bool = False
    similar: Callable = dc.field(default_factory=ClassMatches)


class DummyAttack(Attack):
    def _perturb(self, model, x, y=None, backward_callback=None):
        output, loss_s, grad = self._get_output_and_loss_s_and_grad(model, x, y)
        if backward_callback is not None:
            backward_callback(
                AttackState(x=x, y=y, output=output, x_adv=x, loss_sum=loss_s.item(), grad=grad))
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


# Iterative attacks ################################################################################

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
    elif 'raw':
        return lambda g: g * len(g)


# Attack step updates ##############################################################################

class AttackStepUpdate:
    def __call__(self, delta, grad, x):
        """Updates `delta` and returns the same object (modified)."""
        raise NotImplementedError()

    def init(self, delta, x):
        return delta

    def finish(self, delta, x):
        return delta


class DummyUpdate(AttackStepUpdate):
    def __call__(self, delta, grad, x):
        return delta


# Attacks loop templates and updates (steps) #######################################################


@torch.no_grad()
def zero_grad(delta):
    if delta.grad is not None:
        delta.grad.zero_()


def perturb_iterative(model, x, y, step_count, update, loss, minimize=False, delta_init=None,
                      clip_bounds=(0, 1), similar=None, backward_callback=None):
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
        delta_init (optional): Initial delta.
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
    stop_on_success, loss_fn = similar is not None, loss
    backward_callback = backward_callback or (lambda _: None)

    delta = torch.zeros_like(x) if delta_init is None else delta_init.detach().clone()
    delta.requires_grad_()

    if stop_on_success:  # store arrays because they shrink as attacks succeed
        x_all, delta_all, index = x, delta.detach().clone(), torch.arange(len(x))

    for i in range(step_count):
        x_adv = x + delta
        output = model(x_adv)
        loss = loss_fn(output, y).view(len(x), -1).mean(1).sum()
        zero_grad(delta)
        loss.backward()
        grad = delta.grad  # .detach().clone() unnecessary

        state = AttackState(x=x, y=y, output=output, x_adv=x_adv, loss_sum=loss.item(),
                            grad=grad, step=i)
        backward_callback(state)

        with torch.no_grad():
            if stop_on_success:
                is_adv = similar(state) if minimize else ~similar(state)
                if is_adv.any():  # keep the already succesful adversarial examples unchanged
                    is_nonadv = ~is_adv
                    delta_all[index[is_adv]] = delta[is_adv]
                    x, y, index, grad, delta = (a[is_nonadv] for a in [x, y, index, grad, delta])
                    delta.requires_grad_()
                    if is_adv.all():
                        break  # stop when all adversarial examples are successful

            del loss, output, state  # free some memory
            delta = update(delta, grad.neg_() if minimize else grad, x=x)
            if clip_bounds is not None:
                delta.add_(x).clamp_(*clip_bounds).sub_(x)

    with torch.no_grad():
        if stop_on_success:
            delta_all[index] = delta
            x, delta = x_all, delta_all

        delta = update.finish(delta, x)
    return x + delta


# Attacks ##########################################################################################

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
        return delta.set_(self.project(delta, self.eps))  # TODO: check for p in {1, 2}


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
    grad_preprocessing: str = 'sign'
    rand_init: bool = True
    p: float = np.inf

    def _perturb(self, model, x, y=None, delta_init=None, backward_callback=None):
        if delta_init is None:
            delta_init = (rand_init_delta(x, self.p, self.eps, self.clip_bounds) if self.rand_init
                          else torch.zeros_like(x))
        update = PGDUpdate(step_size=self.step_size, eps=self.eps,
                           grad_preprocessing=self.grad_preprocessing, project_or_p=self.p)

        return perturb_iterative(model, x, y, step_count=self.step_count, update=update,
                                 loss=self.loss, minimize=self.minimize, delta_init=delta_init,
                                 clip_bounds=self.clip_bounds,
                                 similar=self.similar if self.stop_on_success else None,
                                 backward_callback=backward_callback)


@dataclass
class PGDVATAttack(PGDAttack):
    to_virtual_target: Callable = _to_soft_target
    loss: Callable = dc.field(default_factory=KLDivLossWithLogits)


@dataclass
class VATUpdate(AttackStepUpdate):
    eps: float  # final perturbation size
    xi: float  # optimization perturation size
    p: Number

    def __call__(self, delta, grad, x):
        return delta.set_(ops.batch.normalize_by_norm(grad, self.p).mul_(self.xi))

    def finish(self, delta, x):
        return ops.batch.normalize_by_norm(delta, self.p, inplace=True).mul_(self.eps)


@dataclass
class VATAttack(Attack):
    to_virtual_target: Callable = _to_soft_target
    loss: Callable = dc.field(default_factory=KLDivLossWithLogits)

    # TODO: set default CIFAR eps, xi and delta_init
    eps: float = 10  # * 448/32
    xi: float = 1e-6  # * 448/32
    step_count: int = 1
    p: Number = 2

    def _perturb(self, model, x, y=None, backward_callback=None):
        delta_init = rand_init_delta(x, self.p, self.xi, self.clip_bounds)
        update = VATUpdate(xi=self.xi, eps=self.eps, p=self.p)
        return perturb_iterative(model, x, y, step_count=self.step_count, update=update,
                                 loss=self.loss, minimize=self.minimize, delta_init=delta_init,
                                 clip_bounds=self.clip_bounds, backward_callback=backward_callback)


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
    grad_preprocessing: str = 'sign'
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
