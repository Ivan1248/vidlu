from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from vidlu.modules.loss import SoftmaxCrossEntropyLoss
from vidlu.torch_utils import clamp
from vidlu.torch_utils import batchops


# Attack implementations are based on AdverTorch (https://github.com/BorealisAI/advertorch/)

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

    def __init__(self, predict, loss_fn=None, clip_bounds=None,
                 minimize_loss=False, get_predicted_label=_get_predicted_label):
        loss_fn = SoftmaxCrossEntropyLoss(reduction='sum') if loss_fn is None else loss_fn
        self.predict = predict
        self.loss_fn = ((lambda prediction, label: -loss_fn(prediction, label)) if minimize_loss
                        else loss_fn)
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


def rand_init_delta(delta, x, ord, eps, clip_bounds):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside the intersection of a cube and a ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    with torch.no_grad():
        if isinstance(eps, torch.Tensor):
            assert len(eps) == len(delta)

        delta.uniform_(-1, 1) * delta.numel() ** (1 / ord)
        if ord == np.inf:
            delta.mul_(eps)
        elif ord in [1, 2]:
            delta = batchops.restrict_norm(delta, eps, ord)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        return torch.clamp(x + delta, *clip_bounds) - x


def perturb_iterative(x, y, predict, nb_iter, eps, step_size, loss_fn, delta_init=None,
                      ord=np.inf, clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param x: input data.
    :param y: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param step_size: attack step size per iteration.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: (optional float) mininum value per input dimension.
    :param clip_max: (optional float) maximum value per input dimension.
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(x)
    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(x + delta)
        loss = loss_fn(outputs, y)

        loss.backward()
        with torch.no_grad():
            if ord == np.inf:
                grad_sign = delta.grad.sign()
                delta += batchops.mul(grad_sign, step_size, inplace=True)
                delta = delta.clamp_(-eps, eps)  # !!
                delta = (x + delta).clamp_(clip_min, clip_max).sub_(x)  # !!
            elif ord == 2:
                grad = delta.grad
                grad = normalize_by_pnorm(grad)
                delta = delta + batchops.mul(grad, step_size, inplace=True)  # !!
                delta = clamp(x + delta, clip_min, clip_max) - x
                if eps is not None:
                    delta = batchops.restrict_norm(delta, r=eps, p=ord)  # !!
            elif ord == 1:
                assert False
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

            delta.grad.data.zero_()

    x_adv = torch.clamp(x + delta, clip_min, clip_max)
    return x_adv
