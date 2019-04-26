import warnings

import contextlib
from functools import partial

import numpy as np
import torch


@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


@contextlib.contextmanager
def save_grads(params):
    param_grads = [(p, p.grad.clone().detach()) for p in params]
    yield
    for p, g in param_grads:
        p.grad = g


def profile(func, on_cuda=True):
    on_cuda = True
    with torch.autograd.profiler.profile(use_cuda=on_cuda) as prof:
        output = func()
    if on_cuda:
        torch.cuda.synchronize()
    return output, prof.key_averages().table('cuda_time_total')


def to_one_hot(y: torch.Tensor, c: int, dtype=None, device=None):
    y_flat = y.view(-1, 1)
    return (torch.zeros(y_flat.shape[0], c, dtype=dtype, device=device)
            .scatter_(1, y_flat, 1).view(*y.shape, -1))


def clamp(x, min, max):
    if not isinstance(min, torch.Tensor):
        return x.clamp(min, max)
    return torch.min(torch.max(x, min), max)


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


def linear_min_on_p_ball(grad, max_norm, p):
    """Finds the minimum of a function on a `p`-ball with norm `max_norm`.

    Args:
        grad (Tensor): gradient of the linear function.
        max_norm (float or int or Tensor): a number or an array with the same
            number of elements as grad.
        p (int):
    """
    if p == np.inf:
        return grad.sign().mul_(max_norm)
    elif p == 2:
        return (max_norm * grad).div_(grad.norm(p=p))
    elif p == 1:  # only 1 element can be modified in a single update
        sol = torch.zeros_like(grad)
        maxind = torch.argmax(grad.abs())
        sol.view(-1)[maxind] = torch.sign(grad.view(-1)[maxind])
        return sol
    else:
        raise ValueError(f"Frank-Wolfe LMO solving not implemented for p={p}.")
