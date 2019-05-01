from functools import partial, update_wrapper

import numpy as np
import torch
from . import _single as single
from numbers import Real


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
    return (not is_scalar(a) and not is_scalar(b) and len(a.shape) == len(b.shape) and a.shape[0] ==
            b.shape[0] and all(
                are_broadcastable_dimensions(d, e) for d, e in zip(a.shape, b.shape)))


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


def norm(x, p, keep_dims=False):
    norm_ = x.view(x.size(0), -1).norm(p=p, dim=1)
    return redim_as(norm_, x) if keep_dims else norm_


# TODO: remove
'''
def clamp(x, min, max, bounds_are_batches=False, *, inplace=True):
    if not isinstance(min, torch.Tensor):
        return (x.clamp_ if inplace else x.clamp)(min, max)
    min, max = [redim_as(m, x, bounds_are_batches) for m in [min, max]]
    out = x if inplace else None
    return torch.min(torch.max(x, min, out=out), max, out=out)
'''

'''
def project_to_1_ball_old(x, r, r_is_batch=False, *, inplace=False):
    """Batch version, based on https://gist.github.com/daien/1272551"""
    if r_is_batch:
        if len(r.shape) > 2:
            raise ValueError("Not implemented for `r` having more than 2 array dimensions.")
        r = redim_as(r, x, r_is_batch)

    sign = x.sign()
    x = x.abs_() if inplace else x.abs()
    sorted, _ = x.view(x.shape[0], -1).sort(descending=True)
    cssmns = sorted.cumsum(dim=1) - r
    ind = torch.arange(1, sorted.shape[1] + 1, dtype=torch.float)
    pconds = (sorted * ind) > cssmns
    rhos = torch.tensor([ind[cond][-1] for i, cond in enumerate(pconds)])
    thetas = torch.tensor([cssmns[i, cond][-1] for i, cond in enumerate(pconds)]) / rhos
    return x.sub_(redim_as(thetas, x, True)).clamp_(min=0).mul_(sign)
'''


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
    sorted, _ = x.view(x.shape[0], -1).sort(descending=True)
    cssmns = sorted.cumsum(dim=1) - r
    ind = torch.arange(1, sorted.shape[1] + 1, dtype=torch.float)
    pconds = (sorted * ind) > cssmns
    rhos = torch.tensor([ind[cond][-1] for i, cond in enumerate(pconds)])
    thetas = torch.tensor([cssmns[i, cond][-1] for i, cond in enumerate(pconds)]) / rhos
    return x.sub_(redim_as(thetas, x, True)).clamp_(min=0).mul_(sign)


'''
def restrict_norm_old(x, r, p, r_is_batch=False):
    """The projection is ideal for 2-norm and inf-norm but not for others"""
    # TODO: non-scalar r
    if p == np.inf:
        return clamp(x, -r, r, r_is_batch)
    elif p == 2:  # TODO: ellipsoid
        factor = torch.min(redim_as(r, x, r_is_batch) / redim_as(norm(x, p), x), torch.ones(()))
        return x.mul_(factor, x)
    elif p == 1:
        torch.min(x, project_to_1_ball(x, r, inplace=True))  # TODO: optimize
    else:
        raise NotImplementedError(f"Operation not implemented for {p}-norm.")
'''


def restrict_norm(x, r, p):
    """Projects inputs `x` to p-balls with norm(s) `r`.

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
        torch.min(x, project_to_1_ball(x, r))
    else:
        raise NotImplementedError(f"Operation not implemented for {p}-norm.")


def restrict_norm_by_scaling(x, r, p):
    """Divides inputs by their norms and multiples them by `r`.

    Args:
        x: inputs.
        r: p-ball radius/radii.
        p: p-norm p.
    """
    return x * (r / norm(x, p))


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
        assert False, "mul r"
        grad_flat = grad.view(grad.shape[0], -1)
        sol = torch.zeros_like(grad_flat)
        maxinds = torch.argmax(grad.abs().view(grad.shape[0], -1), dim=1)
        for eind, maxind in enumerate(maxinds):
            sol[eind, maxind] = torch.sign(grad_flat[eind, maxind])
        return sol.view(grad.shape)
    else:
        raise ValueError(f"Frank-Wolfe LMO solving not implemented for p={p}.")
