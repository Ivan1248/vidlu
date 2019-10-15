import functools

import numpy as np
import torch
from . import _single as single
from numbers import Real


# Mutual batch broadcastability

def is_scalar(x):
    return isinstance(x, Real) or isinstance(x, torch.Tensor) and len(x.shape) == 0


def are_broadcastable_dimensions(d, e):
    if not isinstance(d, int) and not isinstance(e, int):
        raise ValueError("The dimensions must be integers")
    return d == e or d == 1 or e == 1


def are_broadcastable(a, b):
    number_of_scalars = int(is_scalar(a)) + int(is_scalar(b))
    if number_of_scalars == 2:
        raise ValueError("At least one of the operands has to be a batch. Both are scalars.")
    if number_of_scalars == 1:
        return True
    minlen = min(len(a.shape), len(b.shape))
    return all(
        are_broadcastable_dimensions(d, e) for d, e in zip(a.shape[-minlen:], b.shape[-minlen:]))


def are_comapatible_batches(a, b):
    return (not is_scalar(a) and not is_scalar(b)
            and len(a.shape) == len(b.shape) and a.shape[0] == b.shape[0]
            and all(are_broadcastable_dimensions(d, e) for d, e in zip(a.shape, b.shape)))


# Making batches correctly broadcastable (batch-broadcastable)


def redim_as(this, other, is_batch=True):
    return redim(this, other.shape, is_batch)


def redim(x, batch_shape, is_batch=True):
    """Returns a view of `x` so that it (usually a batch) is broadcastable.

    The new shape for x is
        x.shape[:1] + [1, ...] + x.shape[1:] if `x` is a batch,
                      [1, ...] + x.shape[1:] otherwise

    Args:
        x (Tensor): input.
        batch_shape (tuple): shape that `x` has to be correctly broadcastable
            to.
        is_batch (bool): whether `x` is a batch, in which case the new array
            dimensions have to be inserted after its first dimension.

    Returns:
        A view of `x` with shape (N, 1, .., *)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if is_batch and (len(x.shape) == 0 or x.shape[0] != batch_shape[0]):
        raise ValueError(f"If `is_batch==True`, the dimension 0 of `x` must be the batch size.")

    if len(batch_shape) == len(x.shape):
        return x

    dim_fill = (1,) * (len(batch_shape) - len(x.shape))
    return x.view((x.shape[:1] + dim_fill + x.shape[1:]) if is_batch else (dim_fill + x.shape))


'''
def batch_op(op, x, a):
    """Performs a binary operation `op` over each item of `batch_tensor` with a
    scalar or tensor contained in `a`.

    `a` is first reshaped so that single-item array dimensions are inserted
    after its first array dimension and then broadcast in the operation.

    Args:
        x (Tensor): a Torch array where the first dimension is the batch
            dimension.
        a: a scalar or a Torch array where the first dimension is either
            the batch dimension or 1.
        inplace (bool): if True, `x` will be mulitiplied in-place.
    """
    return op(x, make_broadcastable(a, x.shape, 0))


def batchify(op):
    return partial(batch_op, op)


def _create_inplace_batch_op(op, op_):
    def batch_op_ip(x, a, inplace=False):
        return batch_op(op_ if inplace else op, x, a)

    return batch_op_ip


mul = _create_inplace_batch_op(torch.Tensor.mul, torch.Tensor.mul_)
div = _create_inplace_batch_op(torch.Tensor.div, torch.Tensor.div_)
add = _create_inplace_batch_op(torch.Tensor.add, torch.Tensor.add_)
sub = _create_inplace_batch_op(torch.Tensor.sub, torch.Tensor.sub_)
'''


# Norms and generalized norms

def abs_pow_sum(x, p, keep_dims=False):
    x = x.view(x.size(0), -1)
    psum = (x.abs().pow_(p) if p % 2 == 0 else x.pow(p)).sum(dim=1)
    return redim_as(psum, x) if keep_dims else psum


def norm(x, p, keep_dims=False):
    norm_ = x.view(x.size(0), -1).norm(p=p, dim=1)
    return redim_as(norm_, x) if keep_dims else norm_


def mean_pow(x, p, keep_dims=False):
    """Generalized mean of `abs(x)`."""
    x = x.view(x.size(0), -1)
    pmean = (x.pow(p) if p % 2 == 0 else x.pow(p)).mean(dim=1)
    return redim_as(pmean, x) if keep_dims else pmean


def generalized_mean(x, p, keep_dims=False):
    return mean_pow(x, p, keep_dims).pow_(1 - p)


def mean_pow_abs(x, p, keep_dims=False):
    x = x.view(x.size(0), -1)
    pmean = (x.abs().pow_(p) if p % 2 == 0 else x.pow(p)).mean(dim=1)
    return redim_as(pmean, x) if keep_dims else pmean


def generalized_mean_abs(x, p, keep_dims=False):
    """Generalized mean of `abs(x)`."""
    return mean_pow_abs(x, p, keep_dims).pow_(1 - p)


def l2_distace_sqr(x, y, keep_dims=False):
    return abs_pow_sum(x - y, 2, keep_dims=False)


def project_to_1_ball(x, r, inplace=False):
    """Batch version, based on https://gist.github.com/daien/1272551

    Args:
        x (Tensor): input.
        r (Number or Tensor): the norm of the ball of the inverse weighting of
            vector dimensions if `r` has the same number of array dimensions as
            `x`. The extreme case is that `r` has the same shape as `x` (which is
            currently not implemented).
        inplace:
    """
    sign = x.sign()
    x = x.abs_() if inplace else x.abs()
    sorted_, _ = x.view(x.shape[0], -1).sort(descending=True)
    cssmns = sorted_.cumsum(dim=1) - r
    ind = torch.arange(1, sorted_.shape[1] + 1, dtype=torch.float)
    pconds = (sorted_ * ind) > cssmns
    rhos = x.new_tensor([ind[cond][-1] for i, cond in enumerate(pconds)])
    thetas = x.new_tensor([cssmns[i, cond][-1] for i, cond in enumerate(pconds)]) / rhos
    return x.sub_(redim_as(thetas, x, True)).clamp_(min=0).mul_(sign)


def normalize_by_norm(x, p):
    """Projects inputs `x` to p-sphere with norm(s) `r` with maximum difference with

    Args:
        x: inputs.
        r: p-ball radius/radii.
        p: p-norm p.
    """
    # TODO: non-scalar r
    if p == np.inf:
        return x.sign()
    elif p == 2:  # TODO: make correct for ellipsoids
        return x.div(norm(x, p, keep_dims=True))
    elif p == 1:  # TODO: optimize
        return project_to_1_ball(x, r=1)
    else:
        raise NotImplementedError(f"Operation not implemented for {p}-norm.")


def project_to_p_ball(x, r, p):
    """Projects inputs `x` to p-ball with norm(s) `r` with minimum L2 distance
    with respect to the original value.

    Args:
        x: inputs.
        r: p-ball radius/radii.
        p: p-norm p.
    """
    # TODO: non-scalar r
    if p == np.inf:
        return single.clamp(x, -r, r)
    elif p == 2:  # TODO: make correct for ellipsoids
        return x.mul_(torch.min(r / norm(x, p, keep_dims=True), torch.ones(())))
    elif p == 1:  # TODO: optimize
        return torch.min(x, project_to_1_ball(x, r))
    else:
        raise NotImplementedError(f"Operation not implemented for {p}-norm.")


def scale_to_norm(x, r, p):
    """Divides inputs by their norms and multiples them by `r`.

    Args:
        x: inputs.
        r: p-ball radius/radii.
        p: p-norm p.
    """
    return x * (r / norm(x, p, keep_dims=True))


def linear_min_on_p_ball(grad, r, p=2):
    """Finds the minimum of a function on a `p`-ball with norm `r`.

    Args:
        grad (Tensor): gradient of the linear function (a batch).
        r (Number or Tensor): the norm of the ball, a number or an array
            with the same number of elements as grad items (optionally a batch).
        p (int): p-norm p.
    """
    # TODO: non-scalar r
    if p == np.inf:
        return grad.sign().mul_(r)
    elif p == 2:
        return grad.mul(r / norm(grad, p, keep_dims=True))
    elif p == 1:  # only 1 element can be modified in a single update
        raise NotImplementedError("mul r")
        grad_flat = grad.view(grad.shape[0], -1)
        sol = torch.zeros_like(grad_flat)
        maxinds = torch.argmax(grad.abs().view(grad.shape[0], -1), dim=1)
        for eind, maxind in enumerate(maxinds):
            sol[eind, maxind] = torch.sign(grad_flat[eind, maxind])
        return sol.view(grad.shape)
    else:
        raise ValueError(f"Frank-Wolfe LMO solving not implemented for p={p}.")
