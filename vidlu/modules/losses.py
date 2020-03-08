import torch
from torch import nn
import torch.nn.functional as F

from vidlu import ops
from vidlu.utils.func import class_to_func
from vidlu.utils.torch import batchnorm_stats_tracking_off, save_grads


# Cross entropy ####################################################################################


class NLLLossWithLogits(nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-1):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none')

    def __call__(self, logits, targets):
        return super().__call__(logits, targets)


nll_loss_with_logits = class_to_func(NLLLossWithLogits)


class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction='none')

    def __call__(self, logits, target_probs):
        return super().__call__(torch.log_softmax(logits, 1), target_probs).sum(1)


kl_div_loss_with_logits = class_to_func(KLDivLossWithLogits)


def entropy(logits):
    log_probs = logits.log_softmax(1)
    return -(log_probs.exp() * log_probs).sum(1)


def cross_entropy_loss_with_logits(logits, target_probs):
    return -(target_probs * logits.log_softmax(1)).sum(1)


def reduce_loss(x, batch_reduction: "Literal['sum', 'mean']" = None,
                elements_reduction: "Literal['sum', 'mean']" = None):
    if batch_reduction == elements_reduction and batch_reduction is not None:
        return getattr(torch, batch_reduction)(x)
    if elements_reduction is not None:
        x = getattr(torch, elements_reduction)(x.view(x.shape[0], -1), -1)
    if batch_reduction is not None:
        x = getattr(torch, batch_reduction)(x, 0)
    return x


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
    """
    Carlini-Wagner
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
