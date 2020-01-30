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
    k = dict(device=x.device, dtype=x.dtype)
    mg = torch.meshgrid([torch.linspace(-1, 1, H, **k), torch.linspace(-1, 1, W, **k)])
    base_grid = torch.stack(list(reversed(mg)), dim=-1)
    base_grid = base_grid.expand(N, H, W, 2)

    m = (1 + int(align_corners)) / 2  # (dimension) - (the most a pixel at an edge can mode inside)
    scale = torch.tensor([2 / max(W - m, 1), 2 / max(H - m, 1)], device=x.device, dtype=x.dtype)
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


def tps(theta, ctrl, grid):
    """Evaluates the thin-plate-spline (TPS) surface at locations arranged in a
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
    """Compute a thin-plate-spline grid from parameters for sampling.

    Taken from https://github.com/cheind/py-thin-plate-spline

    Args:
        theta: Nx(T+3)x2 tensor
            Batch size N, T+3 model parameters for T control points in dx and
            dy.
        ctrl: NxTx2 tensor, or Tx2 tensor
            T control points in normalized image coordinates [0..1]
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
    grid = uniform_grid_2d((H, W), with_h_coord=True, h_dim=0, **k).unsqueeze(0).expand(N, H, W, 3)
    z = tps(theta, ctrl, grid)
    return grid[..., 1:].add(z).mul(2).sub_(1)  # [-1,1] range for F.sample_grid


def tps_sparse(theta, ctrl, xy):
    *N, M = xy.shape[:-1]
    grid = torch.cat([xy.new_ones(()).expand(*N, M, 1), xy], dim=-1)
    if xy.dim() == 2:
        grid = grid.expand(theta.shape[0], *grid.shape)
    z = tps(theta, ctrl, grid.view(grid.shape[0], M, 1, 3))
    return xy + z.view(N, M, 2)


@torch.no_grad()
def uniform_grid_2d(shape, low=0., high=1., with_h_coord=False, h_dim=2, dtype=None,
                    device=None):
    # low should be -1.0 for sample_grid
    H, W = shape
    k = dict(device=device, dtype=dtype)
    mg = torch.meshgrid([torch.linspace(low, high, H, **k), torch.linspace(low, high, W, **k)])
    mg = list(reversed(mg))
    if with_h_coord:
        mg.insert(h_dim, mg[0].new_ones(()).expand(mg[0].shape))
    return torch.stack(mg, dim=-1)


if __name__ == '__main__':
    from torchvision.transforms.functional import to_tensor, to_pil_image, resize
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(
        '/home/igrubisic/data/datasets/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png')
    images = [img]

    m = torch.distributions.uniform.Uniform(-.1, .1)

    src = to_tensor(img).unsqueeze(0)

    c_dst = uniform_grid_2d((1, 1)).view(-1, 2)
    size = src.shape

    theta = m.sample((src.shape[0], c_dst.shape[0] + 2, 2)).squeeze(-1)

    grid = tps_grid(theta, torch.tensor(c_dst), size)
    warped = F.grid_sample(src, grid)

    images += [to_pil_image(warped.squeeze())]

    for i, im in enumerate(images):
        plt.subplot(121 + i)
        plt.imshow(im)
    plt.show()
