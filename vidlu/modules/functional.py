import typing as T
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F


class _Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sx = x.sigmoid()
        return (1 - sx).mul_(x).add_(1).mul_(sx).mul_(grad_output)


swish = _Swish.apply


def grid_2d(N, H, W, device=None, dtype=None):
    k = dict(device=device, dtype=dtype)
    mg = torch.meshgrid([torch.linspace(-1, 1, H, **k), torch.linspace(-1, 1, W, **k)],
                        indexing='ij')
    base_grid = torch.stack(list(reversed(mg)), dim=-1)
    return base_grid.expand(N, H, W, 2)


def grid_2d_like(x):
    N, C, H, W = x.shape
    return grid_2d(N, H, W, device=x.device, dtype=x.dtype)


def warp(x, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warps images in an input batch individually with optical flow.

    Args:
        x: a batch of images with shape (N, C, H, W).
        flow: a batch of flows with shape (N, 2, H, W). At the last dimension,
            horizontal offset is at index 0 and vertical at index 1.
        mode (str): interpolation mode to calculate output values
            `'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``
    """
    _, _, H, W = x.shape
    base_grid = grid_2d_like(x)

    m = (1 + int(align_corners)) / 2  # (dimension) - (the most a pixel at an edge can mode inside)
    scale = torch.tensor([2 / max(W - m, 1), 2 / max(H - m, 1)], device=x.device, dtype=x.dtype)
    flow = torch.einsum('nfhw,f->nhwf', flow, scale)

    return F.grid_sample(x, base_grid + flow, mode=mode, padding_mode=padding_mode,
                         align_corners=align_corners)


def warp_ones(flow, mode='bilinear', align_corners=True, binarization_threshold=None):
    result = warp(torch.tensor(1, device=flow.device).expand(flow.shape[0], 3, *flow.shape[2:]),
                  flow, mode=mode, padding_mode='zeros', align_corners=align_corners)
    if binarization_threshold is not None:
        torch.threshold_(result, binarization_threshold)
        result[result < binarization_threshold] = 0
        result[result >= 0] = 1
    return result


def tps(theta, ctrl, grid):
    r"""Evaluates the thin-plate-spline (TPS) surface at locations arranged in a
    grid.

    The TPS surface is a minimum bend interpolation surface defined by a set of
    control points. The function value for a x,y location is given by

        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y +
            \sum_t=0,T theta[t] U(x,y,ctrl[t])

    This method computes the TPS value for multiple batches over multiple grid
    locations for 2 surfaces in one go.

    Taken from https://github.com/cheind/py-thin-plate-spline

    Args:
        theta: N×(T+3)×2 or N×(T+2)×2 array. Batch size N, T+3 or T+2
            (reduced form) model parameters for T control points in dx and dy.
        ctrl: N×T×2 or T×2 array. T control points in normalized image
            coordinates [0..1]
        grid: N×H×W×3 tensor
            Grid locations to evaluate with homogeneous 1 in first coordinate.

    Returns:
        z: NxHxWx2 tensor
            Function values at each grid location in dx and dy.
    """

    N, H, W, T = *grid.shape[:3], ctrl.shape[-2]

    if ctrl.dim() == 2:
        ctrl = ctrl.unsqueeze(0)
    D = (grid[..., None, 1:] - ctrl[:, None, None, ...]).norm(2, dim=-1)
    U = (D ** 2) * torch.log(D + 1e-9)  # NHWT, RBF values

    w, a = theta[:, :-3, :], theta[:, -3:, :]
    if theta.shape[1] == T + 2:
        w = torch.cat((-w.sum(dim=1, keepdim=True), w), dim=1)

    b = torch.bmm(U.view(N, -1, T), w).view(N, H, W, 2)  # NxHxWx2
    z = torch.bmm(grid.view(N, -1, 3), a).view(N, H, W, 2) + b
    return z


def tps_grid(theta, ctrl, size):
    """Computes a thin-plate-spline grid from parameters for sampling.

    Taken from https://github.com/cheind/py-thin-plate-spline

    Args:
        theta: Nx(T+3)x2 tensor
            Batch size N, T+3 model parameters for T control points in dx and
            dy.
        ctrl: NxTx2 tensor, or Tx2 tensor
            T control points in normalized image coordinates from [0, 1]
        size: tuple
            Output grid size as NxCxHxW. C unused. This defines the output image
            size when sampling.

    Returns:
        grid: NxHxWx2 tensor
            Grid suitable for sampling in pytorch containing source image
            locations for each output pixel.
    """
    N, _, H, W = size
    k = dict(device=theta.device, dtype=theta.dtype)
    grid = uniform_grid_2d((H, W), homog_coord=True, homog_dim=0, **k)
    grid = grid.unsqueeze(0).expand(N, H, W, 3)
    z = tps(theta, ctrl, grid)
    return grid[..., 1:].add(z).mul(2).sub(1)  # [-1,1] range for F.sample_grid


def tps_sparse(theta, ctrl, xy):
    *N, M = xy.shape[:-1]
    grid = torch.cat([xy.new_ones(()).expand(*N, M, 1), xy], dim=-1)
    if xy.dim() == 2:
        grid = grid.expand(theta.shape[0], *grid.shape)
    z = tps(theta, ctrl, grid.view(grid.shape[0], M, 1, 3))
    return xy + z.view(N, M, 2)


@torch.no_grad()
def uniform_grid_2d(shape, low=0., high=1., homog_coord=False, homog_dim=2, dtype=None,
                    device=None):
    # low should be -1.0 for sample_grid
    H, W = shape
    if not isinstance(low, tuple):
        low, high = (low,) * 2, (high,) * 2
    k = dict(device=device, dtype=dtype)
    mg = torch.meshgrid(
        [torch.linspace(low[0], high[0], H, **k), torch.linspace(low[1], high[1], W, **k)],
        indexing='ij')
    mg = list(reversed(mg))
    if homog_coord:
        mg.insert(homog_dim, mg[0].new_ones(()).expand(mg[0].shape))
    return torch.stack(mg, dim=-1)  # X/W, Y/H


def grid_2d_points_to_indices(grid_points, shape, grid_low=0., grid_high=1.):
    indices = grid_points.div(grid_high - grid_low).mul_(
        grid_points.new([shape[1] - 1, shape[0] - 1]))
    indices = indices.add_(0.5).long()
    return torch.stack([indices[..., 1], indices[..., 0]], dim=-1)


# Thin plate spline warp ###########################################################################

def linalg_solve(A, B):
    return torch.linalg.solve(A.float(), B.float()).to(A.dtype)


def _tps_fit(c_src_v, lamb=0., reduced=False, eps=1e-6):
    """Fits a 1D thin plate spline and supports batch inputs.

    Based on code from https://github.com/cheind/py-thin-plate-spline
    """

    def d(a, b):
        return a[..., :, None, :2].sub(b[..., None, :, :2]).norm(2, dim=-1)

    def phi(x):
        return x.pow(2).mul(x.abs().add(eps).log())

    k = dict(dtype=c_src_v.dtype, device=c_src_v.device)
    batch_shape, n = c_src_v.shape[:-2], c_src_v.shape[-2]

    P = phi(d(c_src_v, c_src_v))

    Phi = P if lamb == 0 else P + torch.eye(n, **k).unsqueeze(0) * lamb

    C = torch.ones((*batch_shape, n, 3), **k)
    C[..., 1:] = c_src_v[..., :2]

    v = torch.zeros((*batch_shape, n + 3), **k)
    v[..., :n] = c_src_v[..., -1]

    L = torch.zeros((*batch_shape, n + 3, n + 3), **k)
    L[..., :n, :n] = Phi
    L[..., :n, -3:] = C
    L[..., -3:, :n] = C.transpose(-1, -2)

    theta = linalg_solve(L, v.unsqueeze(-1))  # p has structure w,a
    return theta[..., 1:] if reduced else theta


def tps_params_from_points(c_src, c_dst, reduced=False):
    delta = c_dst - c_src

    c_src_dx = torch.cat([c_src, delta[..., 0, None]], dim=-1)
    c_src_dy = torch.cat([c_src, delta[..., 1, None]], dim=-1)

    theta_dx = _tps_fit(c_src_dx, reduced=reduced)
    theta_dy = _tps_fit(c_src_dy, reduced=reduced)

    return torch.cat([theta_dx, theta_dy], -1)


def tps_grid_from_points(c_src, c_dst, size, reduced=False):
    theta = tps_params_from_points(c_src, c_dst, reduced=reduced)
    return tps_grid(theta, c_src, size=size)


def backward_tps_grid_from_points(c_src, c_dst, size, reduced=False):
    """Creates "backward" TPS sampling grid for grid_sample."""
    return tps_grid_from_points(c_dst, c_src, size, reduced=reduced)


# Gaussian forward warp ############################################################################

def gaussian_forward_warp_josa(features, flow, sigma=1., normalize=True):
    epsilon = 1e-6
    B, C, H, W = features.shape
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid_own_coords = torch.cat((xx, yy), 1).float().to(flow.device)  # B, 2, H, W
    grid_own_coords.requires_grad = False
    grid_other_coords = grid_own_coords.clone().view(B, 2, H * W)
    grid_other_coords = grid_other_coords.view(B, 2, H * W, 1, 1).repeat(1, 1, 1, H,
                                                                         W)  # B, 2, H*W, H, W
    grid_other_coords.requires_grad = False
    exact_flow = grid_own_coords.view(B, 2, 1, H, W) - grid_other_coords
    estimated_flow = flow.view(B, 2, H * W, 1, 1).repeat(1, 1, 1, H, W)
    weights = torch.exp(-(((estimated_flow - exact_flow) ** 2) / sigma).sum(1))  # B, H*W, H, W
    future_features = torch.bmm(features.view(B, C, H * W),
                                weights.view(B, H * W, H * W))  # B, C, H*W
    future_features = future_features.view(B, C, H, W)
    if normalize:
        future_features /= (weights.sum(1) + epsilon).unsqueeze(1)
    return future_features


def shuffle(x, dim):
    if not isinstance(dim, T.Sequence):
        dim = [dim]
    return x[[torch.randperm(s) if i in dim else np.s_[:] for i, s in enumerate(x.shape)]]


def qkv(x, Wq, Wk, Wv):
    mm = partial(torch.einsum, 'ntc, cd -> ntd')
    return tuple(mm(x, W) for W in (Wq, Wk, Wv))


def qkv_attention(q, k, v):
    l = torch.einsum('ntd, nud -> ntu', q, k)
    a = F.softmax(l, dim=-1)
    return torch.einsum('ntu, nue -> nte', a, v)


def attention(x, A):
    return torch.einsum('ntu, nuc -> ntc', A, x)


def qkv_self_attention(x, Wq, Wk, Wv):
    q, k, v = qkv(x, Wq, Wk, Wv)
    return qkv_attention(q, k, v)


def multi_head_qkv_attention(q, k, v, Pqs, Pks, Pvs, Wo):
    mm = partial(torch.einsum, 'ntc, cd -> ntd')
    attn_outs = [qkv_attention(mm(q, Pq), mm(k, Pk), mm(v, Pv))
                 for Pq, Pk, Pv in zip(Pqs, Pks, Pvs)]
    return mm(torch.cat(attn_outs, dim=-1), Wo)


def multi_head_qkv_self_attention(x, Pq, Pk, Pv, Pqs, Pks, Pvs, Wo):
    q, k, v = qkv_self_attention(x, Pq, Pk, Pv)
    return multi_head_qkv_attention(q, k, v, Pqs, Pks, Pvs, Wo)
