import typing as T

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from vidlu.utils.func import class_to_func
from vidlu.torch_utils import batchnorm_stats_tracking_off, save_grads
from vidlu.ops import one_hot
from vidlu import metrics
from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj


# Loss adapter #####################################################################################

def logits_to_probs(x):
    return x.softmax(1)


def to_labels(x):
    return x.max(1)[1]


def labels_to_probs(x, c, dtype):
    return one_hot(x, c=c, dtype=dtype)


class LossAdapter:
    def __init__(self, loss,
                 pred: T.Literal[logits_to_probs, to_labels],
                 target: T.Literal[to_labels, labels_to_probs]):
        self.loss, self.pred, self.target = loss, pred, target

    def __call__(self, pred, target):
        return self.loss(self.pred(pred), self.target(target))


# Information-theoretic ############################################################################

"""
Losses that have the "_l" suffix in the name accept 2 arguments:
    1. logits of the approximating distribution
    2. probabilities or integer target labels 

Losses that have the "ll" suffix in the name accept 2 arguments:
    1. logits of the approximating distribution
    2. logits of the target distribution
"""


class NLLLossWithLogits(nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-1):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none')

    def __call__(self, logits, target):
        return super().__call__(logits, target)


nll_loss_l = class_to_func(NLLLossWithLogits)


def nll_loss(probs, target):
    return -torch.log(torch.einsum("nc...,nc...->n...", probs,
                                   F.one_hot(target, probs.shape[1]).transpose(-1, 1)))


def kl_div_l(logits, target):
    if target.dim() == logits.dim() - 1:
        return nll_loss_l(logits, target)
    return F.kl_div(torch.log_softmax(logits, 1), target, reduction='none').sum(1)


def crossentropy_l(logits, target):
    if target.dim() == logits.dim() - 1:
        return nll_loss_l(logits, target)
    return -(target * logits.log_softmax(1)).sum(1)


def crossentropy_ll(logits, target_logits):
    return crossentropy_l(logits, target_logits.softmax(1))


def symmetric_crossentropy_ll(logits, target_logits):
    return crossentropy_ll(logits, target_logits) + crossentropy_ll(target_logits, logits)


def clipped_rev_crossentropy_ll(logits, target_logits, A):
    # https://arxiv.org/abs/1908.06112
    # similar to reverse cross-entropy with smoothed
    return -(logits.softmax() * target_logits.log_softmax(1).clamp(-A).sum(1))


def rev_crossentropy_ll(target_logits, logits):
    return crossentropy_l(logits, target_logits.softmax(1))


def kl_div_ll(logits, target_logits):
    return kl_div_l(logits, target_logits.softmax(1))


def rev_kl_div_ll(target_logits, logits):
    return kl_div_ll(logits, target_logits)


def symmetric_crossentropy_ll(p_logits, q_logits):
    return 0.5 * (crossentropy_l(p_logits, q_logits.softmax(1))
                  + crossentropy_l(q_logits, p_logits.softmax(1)))


def js_div_ll(p_logits, q_logits):
    return 0.5 * (kl_div_l(p_logits, q_logits.softmax(1))
                  + kl_div_l(q_logits, p_logits.softmax(1)))


def entropy_l(logits):
    log_probs = logits.log_softmax(1)
    return -(log_probs.exp() * log_probs).sum(1)


def reduce_loss(x, batch_reduction: T.Literal['sum', 'mean', None] = None,
                elements_reduction: T.Literal['sum', 'mean', None] = None):
    if batch_reduction == elements_reduction and batch_reduction is not None:
        return getattr(torch, batch_reduction)(x)
    if elements_reduction is not None:
        x = getattr(torch, elements_reduction)(x.view(x.shape[0], -1), -1)
    if batch_reduction is not None:
        x = getattr(torch, batch_reduction)(x, 0)
    return x


# Distance losses ##################################################################################

def probs_sqr_l2_dist(probs, target):
    return (probs - target).pow_(2).sum(1)


def probs_sqr_l2_dist_l(logits, target):
    return probs_sqr_l2_dist(logits.softmax(1), target)


def probs_sqr_l2_dist_ll(logits, target_logits):
    return probs_sqr_l2_dist_l(logits, target_logits.softmax(1))


# Confidence thresholding ##########################################################################

def conf_thresh_kl_div_l(logits, target, conf_thresh):
    return kl_div_l(logits, target) * (target.max(1).values >= conf_thresh)


def conf_thresh_kl_div_ll(logits, target_logits, conf_thresh):
    return conf_thresh_kl_div_l(logits, target_logits.softmax(1), conf_thresh=conf_thresh)


def conf_thresh_probs_sqr_l2_dist_ll(logits, target_logits, conf_thresh):
    target = target_logits.softmax(1)
    return probs_sqr_l2_dist_l(logits, target) * (target.max(1).values >= conf_thresh)


def uncertain_kl_div_ll(logits, target_logits):
    target = target_logits.softmax(1)
    targ_sorted = target.sort(descending=True)


# mIoU #############################################################################################

def neg_soft_mIoU_ll(logits, target_logits, batch=True, weights=None):  # TODO
    return neg_soft_mIoU_l(logits, target_logits.softmax(1), batch=batch, weights=weights)


def neg_soft_mIoU_l(logits, target, batch=False, weights=None):  # TODO
    labels = target.dim() == logits.dim() - 1
    pred = logits.softmax(1).transpose(1, -1)
    pred = pred.reshape((pred.shape[0], -1, pred.shape[-1]) if batch else (-1, pred.shape[-1]))
    if labels:  # target contains labels, not probabilities
        target = target.view(pred.shape[:-1])
        cm = metrics.soft_pred_multiclass_confusion_matrix(target, pred)
    else:
        target = target.transpose(1, -1).view(pred.shape)
        cm = metrics.all_soft_multiclass_confusion_matrix(target, pred)
    if weights is not None:
        return -torch.einsum("...ki,i->...", metrics.classification_metrics(cm, 'IoU'), weights)
    return -metrics.classification_metrics(cm, 'mIoU')


# Adversarial training #############################################################################

def _l2_normalize(d, eps=1e-8):
    d_reshaped = d.view(d.shape[0], -1, *([1] * (d.dim() - 2)))
    return d / (torch.norm(d_reshaped, dim=1, keepdim=True) + eps)


class VATLoss(nn.Module):
    # copied from https://github.com/lyakaap/VAT-pytorch/ and modified

    def __init__(self, xi=1e-6, eps=1.0, iter_count=1):  #
        """VAT loss

        Args:
            xi: xi hyperparameter for the finite difference approximation
            eps: perturbation norm
            iter_count: number of iterations for generating perturbations
        """
        super().__init__()
        self.xi = xi
        self.eps = eps
        self.iter_count = iter_count

    def forward(self, model, x, pred=None):
        with batchnorm_stats_tracking_off(model):
            if pred is None:
                with torch.no_grad():
                    pred = F.softmax(model(x), dim=1)

            # prepare random unit tensor
            d = _l2_normalize(torch.rand(x.shape, device=x.device).sub_(0.5))

            def get_kl_div(r):
                pred_hat = model(x + r)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                return F.kl_div(logp_hat, pred, reduction='batchmean')

            # approximate the direction of maximum loss
            with save_grads(model.parameters()):
                for _ in range(self.iter_count):
                    d.requires_grad_()
                    loss = get_kl_div(self.xi * d)
                    loss.backward()
                    d = _l2_normalize(d.grad)
                    model.zero_grad()

            return get_kl_div(x + d * self.eps)


class CarliniWagnerLoss(nn.Module):
    """Carlini-Wagner
    Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self):
        super(CarliniWagnerLoss, self).__init__()

    def forward(self, x, target):
        """
        :param x: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = x.size(1)
        label_mask = F.one_hot(target, num_classes, dtype=torch.float)
        correct_logit = torch.sum(label_mask * x, dim=1)
        wrong_logit = torch.max((1. - label_mask) * x, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + 50.).sum()
        return loss


# Generative

def input_image_nll(x, z, bin_count=256):
    N, dim = len(x), x[0].numel()
    ll_z = -0.5 * (z ** 2 + np.log(2 * np.pi))
    ll_z = ll_z.view(N, -1).sum(-1)
    ll_z -= np.log(bin_count) * dim
    loss_ladj = -Ladj.get(z)()
    loss_ll_z = -ll_z
    return (loss_ladj + loss_ll_z) / dim
