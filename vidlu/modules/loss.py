import torch
from torch import nn
import torch.nn.functional as F

from vidlu import ops
from vidlu.utils.torch import batchnorm_stats_tracking_off, save_grads

# Cross entropy ####################################################################################

NLLLossWithLogits = nn.CrossEntropyLoss


class KLDivLossWithLogits(nn.KLDivLoss):
    def __call__(self, logits, target_probs):
        return super().__call__(torch.log_softmax(logits, 1), target_probs)


# Adversarial training #############################################################################

def _l2_normalize(d, eps=1e-8):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
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
        with batchnorm_stats_tracking_off():
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

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = ops.one_hot(target, num_classes, dtype=torch.float)
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + 50.).sum()
        return loss
