import torch
import torch.nn.functional as F


# Conversion from RGB

def _rgb_to_hcmm(im):
    # https://en.wikipedia.org/wiki/HSL_and_HSV
    R, G, B = (im[..., i, :, :] for i in range(3))

    M, m = im.max(-3)[0], im.min(-3)[0]
    C = M - m  # chroma

    C_div = torch.where(C == 0, torch.tensor(1., device=C.device, dtype=C.dtype), C)
    maxg, maxb = (x == M for x in (G, B))

    H = (G - B).div_(C_div)
    H[maxg] = (B - R).div_(C_div)[maxg].add_(2.)
    H[maxb] = (R - G).div_(C_div)[maxb].add_(4.)
    H[m == M] = 0.0
    H = (H / 6.0) % 1.0

    return H, C, M, m


def rgb_to_hsv(im):
    r"""Converts a RGB image to HSV.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): RGB image.

    Returns:
        torch.Tensor: HSV image.
    """

    H, C, M, m = _rgb_to_hcmm(im)
    S = C / torch.where(M == 0, torch.tensor(1., device=M.device, dtype=M.dtype), M)
    return torch.stack([H, S, M], dim=-3)


def rgb_to_hsl(im):
    r"""Converts a RGB image to HSL.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): RGB image.

    Returns:
        torch.Tensor: HSL image.
    """

    H, C, M, m = _rgb_to_hcmm(im)
    L2 = M.add_(m)  # del M
    S = C.div_(torch.where((L2 == 0) | (L2 == 2), torch.tensor(1.), 1 - (L2 - 1).abs_()))
    L = L2.mul_(0.5)  # del L2
    return torch.stack([H, S, L], dim=1)


def rgb_to_hsi(im):
    r"""Converts a RGB image to HSI.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): RGB image.

    Returns:
        torch.Tensor: HSI image.
    """

    H, C, M, m = _rgb_to_hcmm(im)
    I = im.mean(1)
    S = torch.where(I == 0, torch.tensor(0.), 1 - m.div_(I))
    return torch.stack([H, S, I], dim=1)


def rgb_to_luma(im, rec='709'):
    rec = str(rec)
    if rec not in ['709', '601']:
        raise ValueError("Only Rec. 601 ('601') and Rec. 709 ('709') are supported,")
    weights = torch.tensor([.2126, .7152, .0722] if rec == '709' else [.2989, .5870, .1140])
    return torch.einsum('c, nchw -> nhw', weights, im)


# Conversion to RGB

def _hcm_to_rgb(H, C, M):
    # https://en.wikipedia.org/wiki/HSL_and_HSV
    H6 = 6 * H
    min_ = lambda a, b: torch.min(a, b)  # , out=a)
    one = torch.tensor(1., dtype=C.dtype, device=C.device)

    def f(shift):
        Hs = (H6 + shift) % 6
        # Autograd error if mul_ used isntead of mul
        return M - F.relu(min_(min_(Hs, 4 - Hs), one), inplace=True).mul(C)

    return torch.stack([f(5), f(3), f(1)], dim=1)


def hsv_to_rgb(im):
    r"""Converts an HSV image to RGB.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): HSV image.

    Returns:
        torch.Tensor: RGB image.
    """
    H, S, V = im[:, 0], im[:, 1], im[:, 2]
    return _hcm_to_rgb(H, C=V * S, M=V)


def hsl_to_rgb(im):
    r"""Converts an HSL image to RGB.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): HSL image.

    Returns:
        torch.Tensor: RGB image.
    """
    H, S, L = im[:, 0], im[:, 1], im[:, 2]
    C = (1 - (2 * L).sub_(1).abs_()).mul_(S)
    return _hcm_to_rgb(H, C, M=(C * 0.5).add_(L))


def hsi_to_rgb(im):
    r"""Converts an HSI image to RGB.

    All channels are assumed to have values in [0, 1].

    Args:
        im (torch.Tensor): HSi image.

    Returns:
        torch.Tensor: RGB image.
    """
    H, S, I = im[:, 0], im[:, 1], im[:, 2]
    C = (I * S).mul_(3).div_(2 - (((6 * H) % 2).sub_(1)).abs())
    m = (1 - S).mul_(I)
    return _hcm_to_rgb(H, C, M=C + m)


# might be useful for some other color spaces
def hsv_to_rgb_slow(im):
    r"""Converts an HSY image to RGB.

    All channels are assumed to have values in [0, 1].

    Args:
     im (torch.Tensor): HSY image.

    Returns:
    torch.Tensor: RGB image.
    """
    h, s, v = [im[:, i] for i in range(3)]

    h6 = 6 * h
    hi = torch.floor(h6)
    f = h6 - hi
    p = (1 - s).mul_(v)
    q = (1 - f * s).mul_(v)
    t = (1 - (1 - f).mul_(s)).mul_(v)

    out = torch.stack([hi] * 3, dim=-3) % 6
    triplets = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]

    for i, trip in enumerate(triplets):
        mask = out == i
    out[mask] = torch.stack(trip, dim=-3)[mask]

    return out
