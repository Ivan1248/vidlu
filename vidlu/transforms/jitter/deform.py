# From Marin Oršić
# NOTE: not used.

import torch
import torch.distributions as td
import torch.nn.functional as F
import kornia
from typing import List


def warp_points(pts, M):  # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Applies a homography mapping to point coordinates
    """
    B, N = pts.shape[:2]
    cp = torch.cat([pts, torch.ones((B, N, 1), dtype=pts.dtype, device=pts.device)], dim=-1)
    cp2 = cp.bmm(M)
    return cp2[..., :-1].div(cp2[..., -1:]).reshape(B, N, 2)


class BaseDeform:
    def __call__(self, grid):  # type: (torch.Tensor) -> (torch.Tensor)
        raise NotImplementedError


class CompositeDeform(BaseDeform):
    def __init__(self, deforms: List[BaseDeform]):
        self.deforms = deforms

    def __call__(self, grid):  # type: (torch.Tensor) -> (torch.Tensor)
        for d in self.deforms:
            grid = d(grid)
        return grid


class GridDeform(BaseDeform):
    def __init__(self, dist: td.Distribution = None, points=4, device=torch.device('cpu')):
        if (dist is not None) is False:
            dist = td.uniform.Uniform(torch.tensor(-.05).to(device), torch.tensor(.05).to(device))
        self.dist = dist
        self.points = points

    def __call__(self, grid):
        n, h, w, _ = grid.shape
        p = self.points
        mesh = kornia.create_meshgrid(p, p, normalized_coordinates=False, device=grid.device).div(
            p).add(.5 / p)
        disp = self.dist.sample(mesh.shape).mul_(grid.new([w, h]).view(1, 1, 1, 2))
        return F.interpolate(disp.permute(0, 3, 1, 2), size=(h, w), mode='bicubic',
                             align_corners=True) \
            .permute(0, 2, 3, 1).add(grid)


class HomographyDeform(BaseDeform):
    def __init__(self,
                 dist_ang: td.Distribution = None,
                 dist_trans: td.Distribution = None,
                 dist_scale: td.Distribution = None,
                 device=torch.device('cpu')
                 ):
        if (dist_ang is not None) is False:
            dist_ang = td.uniform.Uniform(torch.tensor(-60.).to(device),
                                          torch.tensor(60.).to(device))
        if (dist_trans is not None) is False:
            dist_trans = td.uniform.Uniform(torch.tensor(.25).to(device),
                                            torch.tensor(.75).to(device))
        if (dist_scale is not None) is False:
            dist_scale = td.uniform.Uniform(torch.tensor(.6).to(device),
                                            torch.tensor(1. / .6).to(device))
        self.dist_ang = dist_ang
        self.dist_trans = dist_trans
        self.dist_scale = dist_scale

    def _find_H(self, size, device):
        # https://stackoverflow.com/a/38423415/5265233
        n, h, w = size
        theta = torch.zeros(n, device=device)
        theta.add_(self.dist_ang.sample(theta.shape)).deg2rad_()
        R = torch.eye(3, device=device).unsqueeze(0).repeat(n, 1, 1)
        T = torch.eye(3, device=device).unsqueeze(0).repeat(n, 1, 1)
        S = torch.eye(3, device=device).unsqueeze(0).repeat(n, 1, 1)
        R[:, 0, 0] = theta.cos()
        R[:, 1, 1] = theta.cos()
        R[:, 0, 1] = theta.sin()
        R[:, 1, 0] = theta.sin().mul_(-1)
        T[:, 0, 2] = self.dist_trans.sample((n,)).mul_(w)
        T[:, 1, 2] = self.dist_trans.sample((n,)).mul_(h)
        S[:, 0, 0] *= self.dist_scale.sample((n,))
        S[:, 1, 1] *= self.dist_scale.sample((n,))
        T2 = T.clone()
        T2[:, :2, 2] *= -1
        H = T.bmm(S.bmm(R.bmm(T2)))
        return H.div(H[..., -1, -1]).transpose(-2, -1)

    def __call__(self, grid):
        n, h, w, _ = grid.shape
        H = self._find_H((n, h, w), grid.device)
        gw = warp_points(grid.view(n, -1, 2), H).view(n, h, w, 2)
        return gw
