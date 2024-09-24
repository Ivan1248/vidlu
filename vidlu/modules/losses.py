import functools
import typing as T

import einops
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typeguard import typechecked

from vidlu.torch_utils import norm_stats_tracking_off, preserve_grads
from vidlu.ops import one_hot
from vidlu import metrics
from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj


# Loss adapter #####################################################################################

def logits_to_probs(x):
    return x.softmax(1)


def to_labels(x):
    return x.max(1)[1]


def labels_to_probs(x, c, dtype):
    return one_hot(x, c, dtype=dtype)


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
    
When targets are segmentations with ignored elements, `reduction="mean"` results
in the loss being averaged over non-ignored elements.

When targets are categorical probabilities and confidence thresholding is used,
all elements are averaged, which might be inconsistent with the segmentation
target behaviour.
"""


def nll_loss_l(logits, target, weight=None, ignore_index=-1,
               reduction: T.Literal["none", "mean", "sum"] = "none"):
    return F.cross_entropy(logits, target, ignore_index=ignore_index, weight=weight,
                           reduction=reduction)


@typechecked
def _apply_reduction(result, reduction: T.Literal["none", "mean", "sum"] = "none", mask=None):
    if mask is not None and reduction == "none":
        raise ValueError(f'reduction="none" is not supported when a mask is provided.')
    if mask is None:
        return result.mean() if reduction == "mean" else result.sum() if reduction == "sum" else \
            result
    else:
        result = result[mask]
        return result.mean() if reduction == "mean" else result.sum()


@typechecked
def nll_loss(probs, target, reduction: T.Literal["none", "mean", "sum"] = "none", ignore_index=-1):
    N, C, *HW = probs.shape
    if ignore_index < 0:
        target_valid = target.clamp(0, C - 1)
    else:
        target_valid = target.clone()
        target_valid[target == ignore_index] = 0
    target_oh = F.one_hot(target_valid, C)
    target_oh = target_oh.to(probs.dtype)
    result = -torch.log(torch.einsum("nc...,n...c->n...", probs, target_oh))
    return _apply_reduction(result, reduction=reduction, mask=target != ignore_index)


@typechecked
def crossentropy(probs, target, reduction: T.Literal["none", "mean", "sum"] = "none"):
    result = (torch.einsum("nc...,nc...->n...", -torch.log(probs), target.transpose(-1, 1)))
    return _apply_reduction(result, reduction=reduction)


def crossentropy_l(logits, target):
    if target.dim() == logits.dim() - 1:
        return nll_loss_l(logits, target)
    return -(target * logits.log_softmax(1)).sum(1)


def crossentropy_ll(logits, target_logits):
    return crossentropy_l(logits, target_logits.softmax(1))


def symmetric_crossentropy_ll(p_logits, q_logits):
    return 0.5 * (crossentropy_ll(p_logits, q_logits) + crossentropy_ll(q_logits, p_logits))


def clipped_rev_crossentropy_ll(logits, target_logits, A):
    # https://arxiv.org/abs/1908.06112
    # similar to reverse cross-entropy with smoothed
    return -(logits.softmax() * target_logits.log_softmax(1).clamp(-A).sum(1))


def rev_crossentropy_ll(target_logits, logits):
    return crossentropy_l(logits, target_logits.softmax(1))


def kl_div_l(logits, target, log_target=False):
    if target.dim() == logits.dim() - 1:
        return nll_loss_l(logits, target, log_target=log_target)
    return F.kl_div(torch.log_softmax(logits, 1), target, reduction='none',
                    log_target=log_target).sum(1)


def kl_div_ll(logits, target_logits):
    return kl_div_l(logits, target_logits.log_softmax(1), log_target=True)


def rev_kl_div_ll(target_logits, logits):
    return kl_div_ll(logits, target_logits)


def js_div_ll(p_logits, q_logits):
    m_log = p_logits.softmax(1).add(q_logits.softmax(1)).div_(2).log_()
    return F.kl_div(m_log, p_logits.log_softmax(1), reduction='none', log_target=True).sum(1).add(
        F.kl_div(m_log, q_logits.log_softmax(1), reduction='none', log_target=True).sum(1)).div_(2)


def entropy_l(logits):
    log_probs = logits.log_softmax(1)
    return -torch.einsum("ij..., ij... -> i...", log_probs.exp(), log_probs)


def reduce_loss(x, batch_reduction: T.Literal['sum', 'mean', None] = None,
                elements_reduction: T.Literal['sum', 'mean', None] = None):
    if batch_reduction == elements_reduction and batch_reduction is not None:
        return getattr(torch, batch_reduction)(x)
    if elements_reduction is not None:
        x = getattr(torch, elements_reduction)(x.view(x.shape[0], -1), -1)
    if batch_reduction is not None:
        x = getattr(torch, batch_reduction)(x, 0)
    return x


# Special supervised ###############################################################################

def downsample_segmap(segmap, factor, ambiguous_value: T.Literal[-1, 'mode']):  # TODO
    from vidlu.modules.components import Baguette
    segmap_bagu = Baguette(factor)(segmap.unsqueeze(1))
    if ambiguous_value == 'mode':
        segmap_ds = segmap_bagu.mode(dim=1).values
    else:
        segmap_ds = segmap_bagu.max(1).values
        segmap_ds[segmap_bagu.min(1).values != segmap_ds] = ambiguous_value
    return segmap_ds


def centroid_supervised_contrastive_loss_z(z, target, temperature=1, centroid_anchor=False,
                                           ignore_index=-1):
    # centroid_anchor=False: pixels attract their centroid and repel other centroids
    # centroid_anchor=True: centroids repel centroids and attract their pixels

    assert z.dim() == 4
    if z.shape[-2:] != target.shape[-2:]:
        target = downsample_segmap(target, target.shape[-2] // z.shape[-2], ambiguous_value=-1)
        # import matplotlib.pyplot as plt; plt.imshow(torch.cat(list(target), dim=-1).cpu(
        # ).numpy()); plt.show()

    masks = torch.stack([(target == i).view(-1)
                         for i in torch.unique(target) if i.item() != ignore_index])  # G(HW)

    z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # NC

    centroids = torch.einsum('gn,nc->gc', masks.float(), z) / masks.sum(1, keepdim=True)  # GC
    groups = [z[m] for m in masks]  # G, *C

    group_pos_scores = [torch.einsum('c,kc->k', centroid, groups[i])
                        for i, centroid in enumerate(centroids)]  # G, *
    group_log_numerators = [s / temperature for s in group_pos_scores]  # G

    if centroid_anchor:
        cent_neg_scores = torch.einsum('gc,hc->gh', centroids, centroids) / temperature  # GG
        torch.diagonal(cent_neg_scores)[:] = 1
        group_to_denominator_neg = cent_neg_scores.exp().sum(1)  # G
        group_to_log_denominators = [(lnn.exp() + group_to_denominator_neg[i]).log()
                                     for i, lnn in enumerate(group_log_numerators)]  # G, *
    else:
        group_to_log_denominators = [
            torch.logsumexp(torch.einsum('hc,gc->hg', zs, centroids) / temperature, dim=1)
            for zs in groups]  # G, *
    group_to_log_ps = [ln - ld
                       for ln, ld in zip(group_log_numerators, group_to_log_denominators)]  # G, *
    log_ps = torch.cat(group_to_log_ps, dim=0)  # (G*)
    return -log_ps.mean()


def centroid_supervised_contrastive_loss_z(z, target, temperature=1, centroid_anchor=False,
                                           ignore_index=-1):
    # centroid_anchor=False: pixels attract their centroid and repel other centroids
    # centroid_anchor=True: centroids repel centroids and attract their pixels

    assert z.dim() == 4
    if z.shape[-2:] != target.shape[-2:]:
        target = downsample_segmap(target, target.shape[-2] // z.shape[-2], ambiguous_value=-1)
        # import matplotlib.pyplot as plt; plt.imshow(torch.cat(list(target), dim=-1).cpu(
        # ).numpy()); plt.show()

    masks = torch.stack([(target == i).view(-1)
                         for i in torch.unique(target) if i.item() != ignore_index])  # G(HW)

    z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # NC

    centroids = torch.einsum('gn,nc->gc', masks.float(), z) / masks.sum(1, keepdim=True)  # GC
    groups = [z[m] for m in masks]  # G, *C

    group_pos_scores = [torch.einsum('c,kc->k', centroid, groups[i])
                        for i, centroid in enumerate(centroids)]  # G, *
    group_log_numerators = [s / temperature for s in group_pos_scores]  # G

    if centroid_anchor:
        cent_neg_scores = torch.einsum('gc,hc->gh', centroids, centroids) / temperature  # GG
        torch.diagonal(cent_neg_scores)[:] = 1
        group_to_denominator_neg = cent_neg_scores.exp().sum(1)  # G
        group_to_log_denominators = [(lnn.exp() + group_to_denominator_neg[i]).log()
                                     for i, lnn in enumerate(group_log_numerators)]  # G, *
    else:
        group_to_log_denominators = [
            torch.logsumexp(torch.einsum('hc,gc->hg', zs, centroids) / temperature, dim=1)
            for zs in groups]  # G, *
    group_to_log_ps = [ln - ld
                       for ln, ld in zip(group_log_numerators, group_to_log_denominators)]  # G, *
    log_ps = torch.cat(group_to_log_ps, dim=0)  # (G*)
    return -log_ps.mean()


# def class_contrastive_loss_z(z, target, temperature=1, ignore_index=-1, sample_count=None):
#     # centroid_anchor=False: pixels attract their centroid and repel other centroids
#     # centroid_anchor=True: centroids repel centroids and attract their pixels
#     assert z.dim() == 4
#     if z.shape[-2:] != target.shape[-2:]:
#         target = downsample_segmap(target, target.shape[-2] // z.shape[-2], ambiguous_value=-1)
#         # import matplotlib.pyplot as plt; plt.imshow(torch.cat(list(target), dim=-1).cpu(
#         ).numpy()); plt.show()
#
#     z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # NC
#     target = target.view(-1)  # N C
#     non_ignore = (target != ignore_index).nonzero().view(-1)
#
#     if sample_count is not None:
#         non_ignore = non_ignore[torch.randperm(len(non_ignore))[: sample_count]]
#
#     z = z[non_ignore]
#     target = target[non_ignore]
#
#     target_repeated = target.expand(*[len(target)] * 2).view(len(target), -1)  # N N
#     different_class = target_repeated.transpose(0, 1) != target  # N N
#
#     similarity = torch.einsum('nc,mc->nm', z, z)
#
#     log_nominators = torch.logsumexp(similarity * (~different_class) / temperature, dim=1)  # N
#     log_denominators = torch.logsumexp(similarity * different_class / temperature, dim=1)  # N
#     log_ps = log_nominators - log_denominators  # N
#     return -log_ps.mean()


# Distance losses ##################################################################################

def probs_sqr_l2_dist(probs, target):
    return (probs - target).pow_(2).sum(1)


def probs_sqr_l2_dist_l(logits, target):
    return probs_sqr_l2_dist(logits.softmax(1), target)


def probs_sqr_l2_dist_ll(logits, target_logits):
    return probs_sqr_l2_dist_l(logits, target_logits.softmax(1))


# Confidence thresholding ##########################################################################

def conf_thresh(loss, kind):
    if kind != 'class-probs':
        raise NotImplementedError()

    @functools.wraps(loss)
    def wrapper(logits, target, conf_thresh, **kwargs):
        return loss(logits, target, **kwargs) * (target.max(1).values >= conf_thresh)

    return wrapper


def conf_thresh_kl_div_l(logits, target, conf_thresh):
    """

    TODO: add option for averaging only over masked pixels
    French averages over all pixels, not just the masked ones. We do so too,
    but it might be inconsistent with `nll_loss_l`, which averages over
    non-ignored pixels (when given segmentation targets).

    Args:
        logits:
        target:
        conf_thresh:

    """
    return kl_div_l(logits, target) * (target.max(1).values >= conf_thresh)


def conf_thresh_kl_div_ll(logits, target_logits, conf_thresh):
    return conf_thresh_kl_div_l(logits, target_logits.softmax(1), conf_thresh=conf_thresh)


def conf_thresh_probs_sqr_l2_dist_ll(logits, target_logits, conf_thresh):
    target = target_logits.softmax(1)
    return probs_sqr_l2_dist_l(logits, target) * (target.max(1).values >= conf_thresh)


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


# Unsupervised losses $#############################################################################

def _check_cross_correlation_inputs(a, b):
    if any([len(s_a := tuple(a.shape)) != 2, len(s_b := tuple(b.shape)) != 2]):
        raise RuntimeError(f"The inputs must be 2-D arrays, but have shapes {s_a} and {s_b}.")
    if len(a) != len(b):
        raise RuntimeError(f"Input sizes ({len(a)}, {len(b)}) do not match.")


def cross_correlation(a, b, bias=False):
    _check_cross_correlation_inputs(a, b)  # N×C
    return (a.T @ b).div_(len(a) - 1 + int(bias))


def _check_cov_input(x):
    if x.ndim != 2:
        raise RuntimeError(f"The input must be a 2-D array, but has the shape {tuple(x.shape)}.")


def cov(x, bias=False):
    _check_cov_input(x)  # N×C
    x_c = x - x.mean(0, keepdim=True)
    return (x_c.T @ x_c).div_(len(x_c) - 1 + int(bias))


def normalized_identity_cross_correlation_l2_loss(a, b, lamb=1):
    """Loss used in Barlow Twins, https://github.com/facebookresearch/barlowtwins"""
    a, b = map(to_batch_of_vectors, (a, b))

    a_norm, b_norm = [(x - x.mean(0)) / x.std(0) for x in [a, b]]  # NxD
    return cross_correlation_square_l2_loss(a_norm, b_norm, lamb=lamb)


def cross_correlation_square_l2_loss(a, b, lamb=1):
    c = cross_correlation(a, b, bias=True)  # empirical feature cross-correlation matrix (C×C)
    c.diagonal().sub_(1)
    c.pow_(2)
    return c.diagonal().sum() + lamb * _off_diagonal(c).sum()


def vicreg_var_loss(x, var_thresh=1):
    x = to_batch_of_vectors(x)
    return torch.relu_(var_thresh - torch.std(x, dim=0, unbiased=True)).mean()


def vicreg_cov_loss(x):
    x = to_batch_of_vectors(x)
    return _off_diagonal(cov(x).pow_(2)).sum().div_(x.shape[1])


def vicreg_loss(a, b, var_thresh=1, coeff_v=25, coeff_i=25, coeff_c=1):
    a, b = map(to_batch_of_vectors, (a, b))

    v_a, v_b = [vicreg_var_loss(x, var_thresh=var_thresh) for x in [a, b]]
    i = F.mse_loss(a, b, reduction="mean")
    c_a, c_b = map(vicreg_cov_loss, [a, b])
    return coeff_v * (v_a + v_b) + coeff_i * i + coeff_c * (c_a + c_b)


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
        with norm_stats_tracking_off(model):
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
            with preserve_grads(model.parameters()):
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
    """Computes the log-probability that each dimension of x belongs ta the corresponding bin out of
     255.

     Assumes that all input elements are between 0 and 1."""
    N, dim = len(x), x[0].numel()

    ll_z = -0.5 * (z ** 2 + np.log(2 * np.pi))
    ll_z = ll_z.view(N, -1).sum(-1)
    ll_z -= np.log(bin_count) * dim

    nladj = -Ladj.get(z)()
    nll_z = -ll_z
    nll = nladj + nll_z
    return nll / dim

# Utilities

def to_batch_of_vectors(x: torch.Tensor):
    """Turns an (N, C, S1, S2, ...) array into a batch of vectors.

    Args:
        x: An (N, C, S1, S2, ...) array.


    Returns:
        Transposed and reshaped array with shape (N S1 S2 ..., C).
    """
    if x.ndim == 2:
        return x
    return einops.rearrange(x.view(*x.shape[:2], -1), 'n c s -> (n s) c')


def _off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
