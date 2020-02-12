import warnings
import math

import numpy as np
import torch


def one_hot(indices: torch.Tensor, c: int, dtype=None):
    """Returns a one-hot array.

    Args:
        indices (Tensor): labels.
        c (int): number of classes.
        dtype: PyTorch data-type.

    Returns:
        A one-hot array with the last array dimension of size `c` and ones at
        `indices`.
    """
    if dtype is torch.int64:
        warnings.warn("You can use one_hot from torch.nn.functional instead.")
    y_flat = indices.view(-1, 1)
    return (torch.zeros(y_flat.shape[0], c, dtype=dtype, device=indices.device)
            .scatter_(1, y_flat, 1).view(*indices.shape, -1))


def clamp(x, min, max, inplace=False):
    """A `clamp` that works with `min` and `max` being arrays.

    Args:
        x: input array.
        min (Tensor or float): lower bound.
        max (Tensor or float): upper bound.

    Returns:
        An array with values bounded with `min` and `max`.
    """
    if not isinstance(min, torch.Tensor):
        return (x.clamp_ if inplace else x.clamp)(min, max)
    out = x if inplace else None
    return torch.min(torch.max(x, min, out=out), max, out=out)


def random_uniform_(x, from_, to, generator=None):
    """Like `torch.tensor.unifom_` but works with `from_` and `to` being a pair
    of arrays describing a hypercube.

    Args:
        x: input array.
        from_ (Tensor or float): lower bound.
        to (Tensor or float): upper bound.

    Returns:
        An array with values bounded with `min` and `max`.
    """
    if not isinstance(from_, torch.Tensor):
        return x.uniform_(from_, to, generator=generator)
    return x.uniform_(generator=generator).mul_(to - from_).add_(from_)


def project_to_1_ball(x, max_norm=1, inplace=False):
    """from https://gist.github.com/daien/1272551"""
    sign = x.sign()
    x = x.abs_() if inplace else x.abs()
    xs, _ = x.view(-1).sort(descending=True)
    cssmn = xs.cumsum(dim=0) - max_norm
    ind = torch.arange(1, xs.shape[0] + 1, dtype=torch.float)
    pcond = (xs * ind) > cssmn
    rho = ind[pcond][-1]  # number of > 0 elements
    theta = cssmn[pcond][-1] / rho
    return x.sub_(theta).clamp_(min=0).mul_(sign)


def project_to_1_ball_i(x, max_norm=1, max_iter=20, eps=1e-6, inplace=False):
    # a bit slower than the algorithm with sorting
    sign = x.sign()
    x = x.abs_() if inplace else x.abs()
    norm = None
    factor = torch.tensor(1.)
    for i in range(max_iter):
        lastnorm = norm
        norm = x.relu().sum()  # 2n
        if i > 0 and norm != 0:
            factor *= 0.5 + 0.5 * 2 * (lastnorm - max_norm) / (2 * lastnorm - norm - max_norm)
        d = abs(norm - max_norm)
        if d < (eps * max_norm):
            break
        x -= (norm - max_norm) * (factor / (1 + float((x > 0).sum())))  # 2n
    else:
        warnings.warn(f"The 1-norm projection did not converge in {max_iter} iterations.")
    return x.clamp_(min=0).mul_(sign)


def linear_min_on_p_ball(x, max_norm, p):
    """Finds the minimum of a function on a `p`-ball with norm `max_norm`.

    Args:
        x (Tensor): gradient of the linear function.
        max_norm (float or int or Tensor): a number or an array with the same
            number of elements as grad.
        p (int):
    """
    if p == np.inf:
        return x.sign().mul_(max_norm)
    elif p == 2:
        return (max_norm * x).div_(x.norm(p=p))
    elif p == 1:  # only 1 element can be modified in a single update
        sol = torch.zeros_like(x)
        maxind = torch.argmax(x.abs())
        sol.view(-1)[maxind] = torch.sign(x.view(-1)[maxind])
        return sol
    else:
        raise ValueError(f"Frank-Wolfe LMO solving not implemented for p={p}.")


# Elementwise functions

def scaled_tanh(x, min=-1., max=1., input_scale=1):
    y_scale, y_offset = 0.5 * (max - min), 0.5 * (max + min)
    return torch.tanh(x if input_scale == 1 else input_scale * x) * y_scale + y_offset


def unscaled_atanh(x, min=-1., max=1., input_scale=1):
    y_scale, y_offset = 0.5 * (max - min), 0.5 * (max + min)
    return torch.tanh(x if input_scale == 1 else input_scale * x) * y_scale + y_offset


def atanh(x, inplace=False):
    den = 1 - x
    return (x.add_(1) if inplace else x + 1).log().div_(den).mul_(0.5)


def harder_tanh(x, h=1, inplace=False):
    """A generalization of tanh that converges to tanh for h → +0 and
    x ↦ min(max(x, 0), 1) for h → ∞, probably"""
    if h == 0:
        return x.tanh_() if inplace else x.tanh()
    euh, elh = math.exp(h), math.exp(-h)
    s = (1 + euh) * (1 + elh) / (euh - elh)  # to make f'(0) = 1
    hs = h * s
    exhs = (x.mul_ if inplace else x.mul)(hs).exp_()
    den, nom = exhs + euh, exhs.add_(elh),
    return nom.div_(den).log().div_(h).add_(1)


def soft_clamp(x, min_, max_, eps, inplace=False):
    """A piecewise continous function consisting of 2 scaled tanh parts and an
    id part from `eps` to `1 - eps`.

    The left limit of the function is 0. The right limit is 1. The function is identity from
    `eps` to `1 - eps`.
    """
    if not inplace:
        x = x.clone()
    l, h = min_ + eps, max_ - eps
    high = x > h
    # ASutograd error if mul_ used instead of mul
    if len(high) > 0:
        x[high] = x[high].sub_(h).div_(eps).tanh_().mul(eps).add_(h)
    low = x < l
    if len(low) > 0:
        x[low] = x[low].sub_(l).div_(eps).tanh_().mul(eps).add_(l)
    return x
