from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import einops as eo

# noinspection PyUnresolvedReferences
import _context
import vidlu.modules.functional as vmf


def flow2rgb(flow):
    hsv = np.zeros(list(flow.shape[:-1]) + [3], dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


img = Image.open('image.jpg')
images = [img]
timages = []
x = to_tensor(img).unsqueeze(0)
try:
    x = x.cuda()
except:
    pass

with torch.no_grad():
    c_src = vmf.uniform_grid_2d((2, 2)).view(-1, 2).unsqueeze(0).to(x.device)
    offsets = c_src * 0
    offsets[..., 0, 0] = np.random.uniform(0, 0.9)
    offsets[..., 0, 1] = np.random.uniform(0, 0.9)

N, C, H, W = x.shape
k = dict(device=x.device, dtype=x.dtype)
mg = torch.meshgrid([torch.linspace(-1, 1, H, **k), torch.linspace(-1, 1, W, **k)])
base_grid = torch.stack(list(reversed(mg)), dim=-1)
base_grid = base_grid.expand(N, H, W, 2).to(x.device)
# embed()

gridfw = vmf.tps_grid_from_points(c_src, c_src + offsets, size=x.shape)
flow = (gridfw - base_grid).permute(0, 3, 1, 2) * gridfw.new(
    [W / 2, H / 2]).view(1, 2, 1, 1)
x_warped = vmf.gaussian_forward_warp_josa(x, flow, sigma=0.3)
# from vidlu.libs.softmax_splatting import softsplat
# x_warped = softsplat.FunctionSoftsplat(x, flow.contiguous(), tenMetric=None, strType="average")

# flow_img = to_tensor(flow2rgb((gridfw - base_grid)[0].numpy()))
timages += [x_warped]
timages += [F.grid_sample(x_warped, gridfw).squeeze_(1)]


def invert_offset_grid(grid_fw, grid_bw):
    grid_nchw = eo.rearrange(grid_fw, "n h w c -> n c h w")
    zero_offset_grid = vmf.uniform_grid_2d(grid_nchw.shape[-2:], low=-1, high=1, device=x.device) \
        .unsqueeze_(0)
    grid_warped = eo.rearrange(F.grid_sample(grid_nchw, grid_bw), "n c h w -> n h w c")
    return zero_offset_grid - (grid_warped - zero_offset_grid)


gridbw = vmf.backward_tps_grid_from_points(c_src, c_src + offsets, size=x.shape)
x_warped = F.grid_sample(x, gridbw).squeeze_(1)
# timages += [x_warped]
# timages += [
#    vmf.gaussian_forward_warp_josa(x_warped.clone(), (gridbw - base_grid).permute(0, 3, 1, 2) * gridbw.new(
#        [W / 2, H / 2]).view(1, 2, 1, 1), sigma=0.5)]
# timages += [F.grid_sample(x_warped, invert_offset_grid(gridfw, gridbw)).squeeze_(1)]

# perturbation module

from vidlu.modules.inputwise import TPSWarp

warp = TPSWarp(forward=True, control_grid_shape=(2, 2), control_grid_align_corners=True, align_corners=False)
x_warped = warp(x)
with torch.no_grad():
    warp.offsets.add_(offsets)
x_warped = warp(x)
timages += [x_warped]
timages += [warp.inverse(x_warped)]

# unrelated - Gaussian
from vidlu.modules.components import GaussianFilter2D

gridrand = torch.randn_like(base_grid) * 6
gridrand = GaussianFilter2D(sigma=20, padding_mode='reflect')(gridrand.permute(0, 3, 1, 2)).permute(
    0, 2, 3, 1)
# timages += [F.grid_sample(x, base_grid + gridrand).squeeze_(1)]

for x in timages:
    r = 1
    for c in vmf.grid_2d_points_to_indices(c_src + offsets, x.shape[-2:])[0]:
        x[..., c[0] - r: c[0] + r, c[1] - r:c[1] + r] = 1.

images += [to_pil_image(x.squeeze().cpu()) for x in timages]

fig, axs = plt.subplots(len(images))
for i, im in enumerate(images):
    axs[i].imshow(im)
plt.show()
