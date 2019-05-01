from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from vidlu.modules.loss import SoftmaxCrossEntropyLoss
from vidlu.torch_utils import clamp, to_one_hot
from vidlu.torch_utils import batchops as B
from vidlu.torch_utils import random


# Attack implementations are based on AdverTorch (https://github.com/BorealisAI/advertorch)

def _get_predicted_label(predict, x):
    """Compute predicted labels given `x`. Used to prevent label leaking

    Args:
        predict: a model.
        x (Tensor): an input for the model.

    Returns:
        A tensor containing predicted labels.
    """
    with torch.no_grad():
        out = predict(x)
        _, y = torch.max(out, dim=1)
        return y


def _get_predicted_label_soft(predict, x, temperature=1):
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
    with torch.no_grad():
        out = predict(x)
        if temperature != 1:
            out /= temperature
        return F.softmax(out, dim=1)


class Attack:
    """Adversarial attack base class.

    Args:
        predict: a model.
        loss_fn: a loss function that is increased by perturbing the input.
        clip_interval (Tuple[float, float]): a tuple representing the
            interval in which input values are valid.
        minimize_loss (bool): If `True`, by calling `perturb`, the
            input is perturbed so that the loss decreases. Otherwise (default),
            the input is perturbed so that the loss increases. It should be set
            to `True` for targeted attacks.
        get_predicted_label ((Tensor -> Tensor), Tensor -> Tensor): a function
            that¸accepts a model and an input, and returns the predicted label
            that can be used as the second argument in `loss_f`. Default: a
            function that returns a (hard) classifier prediction -- argmax over
            dimension 1 of the model output.

    Note:
        If `minimize_loss=True`, the base class constructor wraps `loss_fn` so
        that the loss is multiplied by -1 so that subclasses thus don't have to
        check whether they need to maximize or minimize loss.
        When calling the base constructor, `clip_interval` should be set to
        `None` (default) if the output of `_predict` doesn't have to be clipped.
    """

    def __init__(self, predict, loss_fn=None, clip_bounds=None, minimize_loss=False,
                 get_predicted_label=_get_predicted_label):
        loss_fn = SoftmaxCrossEntropyLoss(reduction='sum') if loss_fn is None else loss_fn
        self.predict = predict
        self.loss_fn = (
            (lambda prediction, label: -loss_fn(prediction, label)) if minimize_loss else loss_fn)
        self.clip_bounds = clip_bounds
        self.get_predicted_label = get_predicted_label

    def perturb(self, x, y=None):
        """Generates an adversarial example.

        Args:
            x (Tensor): input tensor.
            y (Tensor): label tensor. If `None`, the prediction obtained from
                `get_predicted_label(predict, x)` is used as the label.

        Return:
            Perturbed input.
        """
        x_adv = self._perturb(x.detach().clone(),
                              (self.get_predicted_label(x) if y is None else y).detach().clone())
        return x_adv if self.clip_bounds is None else self._clip(x_adv)

    def _get_output_and_loss_and_grad(self, x, y):
        x.requires_grad_()
        output = self.predict(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        return output, loss, x.grad.detach()

    def _clip(self, x):
        return torch.clamp(x, *self.clip_bounds)

    def _perturb(self, x, y):
        # has to be overriden in subclasses
        raise NotImplementedError()


class GradientSignAttack(Attack):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, predict, loss_fn=None, clip_bounds=None, minimize_loss=False,
                 get_predicted_label=_get_predicted_label, eps=0.3):
        """Create an instance of the GradientSignAttack.

        Args:
            eps (float or Tensor): attack step size.
        """
        super().__init__(predict, loss_fn, clip_bounds, minimize_loss, get_predicted_label)
        self.eps = eps

    def perturb(self, x, y=None):
        output, loss, grad = self._get_output_and_loss_and_grad(x, y)
        grad_sign = grad.sign()
        return x + self.eps * grad_sign


def rand_init_delta(x, p, eps, bounds, batch=False):
    """Generates a random perturbation from a unit p-ball scaled by eps.

    Args:
        x (Tensor): input.
        p (Real): p-ball p.
        eps (Real or Tensor):
        bounds (Real or Tensor): clipping bounds.
        batch: whether `x` is a batch.

    """
    with torch.no_grad():
        if batch:
            delta = torch.stack(
                [random.uniform_sample_from_p_ball(p, x.shape[1:]) for _ in range(x.shape[0])])
        else:
            delta = random.uniform_sample_from_p_ball(p, x.shape)
        delta = delta.mul_(eps)
        return torch.clamp(x + delta, *bounds) - x


def perturb_iterative(x, y, predict, iter_count, eps, step_size, loss_fn, delta_init=None, p=np.inf,
                      clip_bounds=(0, 1)):
    """Iteratively maximizes the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    Args:
        x: inputs.
        y: input labels.
        predict: forward pass function.
        iter_count: number of iterations.
        eps: maximum distortion.
        step_size: attack step size per iteration.
        loss_fn: loss function.
        delta_init (optional): tensor contains the random initialization.
        p (optional): the order of maximum distortion (inf or 2).
        clip_bounds (optional): mininum and maximum value pair.

    Returns:
        The perturbed input.
    """

    delta = torch.zeros_like(x) if delta_init is None else delta_init
    delta.requires_grad_()
    for ii in range(iter_count):
        outputs = predict(x + delta)
        loss = loss_fn(outputs, y)

        loss.backward()
        with torch.no_grad():
            if p == np.inf:  # try with restrict_norm instead of sign, mul
                delta += delta.grad.sign().mul_(step_size)
                if eps is not None:
                    delta = B.restrict_norm(delta, eps, p=p)
                delta = (x + delta).clamp_(*clip_bounds).sub_(x)
            elif p in [1, 2]:  # try with restrict_norm_by_scaling instead of restrict_norm
                delta += B.restrict_norm(delta.grad, 1, p=p).mul_(step_size)
                delta = (x + delta).clamp_(*clip_bounds).sub_(x)
                if eps is not None:
                    delta = B.restrict_norm(delta, eps, p=p)  # !!
            else:
                raise NotImplementedError(f"Not implemented for p = {p}.")

            delta.grad.zero_()

    return torch.clamp(x + delta, *clip_bounds)


class PGDAttack(Attack):
    def __init__(self, predict, loss_fn=None, clip_bounds=None, minimize_loss=False,
                 get_predicted_label=_get_predicted_label, eps=0.3, iter_count=40, step_size=0.01,
                 rand_init=True, p=np.inf):
        """The PGD attack (Madry et al, 2017).

        The attack performs nb_iter steps of size eps_iter, while always staying
        within eps from the initial point.
        Paper: https://arxiv.org/pdf/1706.06083.pdf

        Args:
            eps: maximum distortion.
            iter_count: number of iterations.
            step_size: attack step size.
            rand_init (bool, optional): random initialization.
            p (optional): the order of maximum distortion (inf or 2).
        """
        super().__init__(predict, loss_fn, clip_bounds, minimize_loss, get_predicted_label)
        self.eps = eps
        self.iter_count = iter_count
        self.step_size = step_size
        self.rand_init = rand_init
        self.p = p

    def _perturb(self, x, y=None):
        delta = (rand_init_delta(x, self.p, self.eps,
                                 self.clip_bounds) if self.rand_init else torch.zeros_like(x))

        return perturb_iterative(x, y, self.predict, iter_count=self.iter_count, eps=self.eps,
                                 step_size=self.step_size, loss_fn=self.loss_fn, p=self.p,
                                 clip_bounds=self.clip_bounds, delta_init=delta)



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
    def carlini_loss(logits, y):
        y_onehot = to_one_hot(y, logits.shape[-1])
        real = (y_onehot * logits).sum(dim=-1)

        other = ((1.0 - y_onehot) * logits - (y_onehot * TARGET_MULT)).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if targeted:
            loss1 = (other - real + confidence_threshold).relu_()
        else:
            loss1 = (real - other + confidence_threshold).relu_()
        loss2 = (l2distsq).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss


class CarliniWagnerL2Attack(Attack):
    def __init__(self, predict, loss_fn=None, clip_bounds=None, minimize_loss=False,
                 get_predicted_label=_get_predicted_label, num_classes, confidence=0,
                 learning_rate=0.01,
                 binary_search_steps=9, max_iter=10000, abort_early=True, initial_const=1e-3):
        """The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644

        Args:
            num_classes: number of clasess.
            confidence: confidence of the adversarial examples.
            learning_rate: the learning rate for the attack algorithm
            binary_search_steps: number of binary search times to find the optimum
            max_iterations: the maximum number of iterations
            abort_early: if set to true, abort early if getting stuck in local min
            initial_const: initial value of the constant c
        """
        if loss_fn is not None:
            raise NotImplementedError("The CW attack currently does not support a different loss"
                                      " function other than the default. Setting loss_fn manually"
                                      " is not effective.")

        super().__init__(predict, loss_fn, clip_bounds, minimize_loss, get_predicted_label)

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP

    def _loss_fn(self, output, y_onehot, l2distsq, const):
        # TODO: move this out of the class and make this the default loss_fn
        #   after having targeted tests implemented
        real = (y_onehot * output).sum(dim=1)

        # TODO: make loss modular, write a loss class
        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.minimize_loss:
            loss1 = clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = clamp(real - other + self.confidence, min=0.)
        loss2 = (l2distsq).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label

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

        return is_successful(pred, label, self.targeted)

    def _forward_and_update_delta(self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):

        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        output = self.predict(adv)
        l2distsq = calc_l2distsq(adv, transimgs_rescale)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, output.data, adv.data

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min), min=self.clip_min,
                       max=self.clip_max) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

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

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iter):
                loss, l2distsq, output, adv_img = self._forward_and_update_delta(optimizer, x_atanh,
                                                                                 delta, y_onehot,
                                                                                 loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iter // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                self._update_if_smaller_dist_succeed(adv_img, y, output, l2distsq, batch_size,
                                                     cur_l2distsqs, cur_labels, final_l2distsqs,
                                                     final_labels, final_advs)

            self._update_loss_coeffs(y, cur_labels, batch_size, loss_coeffs, coeff_upper_bound,
                                     coeff_lower_bound)

        return final_advs

