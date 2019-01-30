import contextlib

import torch
from torch import nn
import torch.nn.functional as F

from vidlu.utils.torch import save_grads, _disable_tracking_bn_stats


def _l2_normalize(d, eps=1e-8):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + eps
    return d


class VATLoss(nn.Module):
    # from https://github.com/lyakaap/VAT-pytorch/

    def __init__(self, xi=10.0, eps=1.0, iteration_count=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param iteration_count: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = iteration_count

    def forward(self, model, x, pred=None):
        if pred is None:
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        def get_loss(r_adv):
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            return F.kl_div(logp_hat, pred, reduction='batchmean')

        with _disable_tracking_bn_stats(model):
            with save_grads(model):
                for _ in range(self.ip):
                    d.requires_grad_()
                    lds = get_loss(self.xi * d)
                    lds.backward()
                    d = _l2_normalize(d.grad)
                    model.zero_grad()

            # compute LDS
            r_adv = d * self.eps
            lds = get_loss(x + r_adv)
        return lds
