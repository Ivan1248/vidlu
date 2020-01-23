import torch
from torch import nn
from torch.nn import functional as F


def dimensional_function(f_list, *args, **kwargs):
    return f_list[len(args[0].size()) - 3](*args, **kwargs)


def adaptive_avg_pool(x, output_size):
    return dimensional_function(
        [F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d], x, output_size)


def avg_pool(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return dimensional_function(
        [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d], x, kernel_size=kernel_size, stride=stride,
        padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)


def global_avg_pool(x):
    return adaptive_avg_pool(x, 1).squeeze()


def warp(x, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
    """ Warps images in an input batch individually with optical flow.

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
    N, C, H, W = x.shape
    mg = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
    base_grid = torch.stack(list(reversed(mg)), dim=-1)
    base_grid = base_grid.expand(N, H, W, 2)

    m = (1 + int(align_corners)) / 2  # (dimension) - (the most a pixel at an edge can mode inside)
    scale = torch.tensor([2 / max(W - m, 1), 2 / max(H - m, 1)], device=x.device)
    flow = torch.einsum('nfhw,f->nhwf', flow, scale)

    # PWC-Net has this mask multiplied with the output
    # mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    # mask = nn.functional.grid_sample(mask, base_grid + flow)
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    return F.grid_sample(x, base_grid + flow, mode=mode, padding_mode=padding_mode,
                         align_corners=align_corners)


def warp_ones(flow, mode='bilinear', align_corners=True):
    return warp(torch.tensor(1, device=flow.device).expand(flow.shape[0], 3, *flow.shape[2:]), flow,
                mode=mode, padding_mode='zeros', align_corners=align_corners)
