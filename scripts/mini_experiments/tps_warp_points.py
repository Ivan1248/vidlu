from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
import numpy as np

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
timages=[]
x = to_tensor(img).unsqueeze(0)

with torch.no_grad():
    c_src = vmf.uniform_grid_2d((2, 2)).view(-1, 2).unsqueeze(0)
    offsets = c_src * 0
    offsets[..., 0, 0] = 0.5
    offsets[..., 0, 1] = 0.3

forward = False

gridfw = vmf.tps_grid_from_points(c_src, c_src + offsets, size=x.shape)

N, C, H, W = x.shape
k = dict(device=x.device, dtype=x.dtype)
mg = torch.meshgrid([torch.linspace(-1, 1, H, **k), torch.linspace(-1, 1, W, **k)])
base_grid = torch.stack(list(reversed(mg)), dim=-1)
base_grid = base_grid.expand(N, H, W, 2)
# embed()
x_warped = vmf.gaussian_forward_warp_josa(x, (gridfw - base_grid).permute(0, 3, 1, 2) * gridfw.new([W / 2, H / 2]).view(1, 2, 1, 1), sigma=0.5)
flow_img = to_tensor(flow2rgb((gridfw - base_grid)[0].numpy()))
timages += [x_warped]

gridbw = vmf.backward_tps_grid_from_points(c_src, c_src + offsets, size=x.shape)
x_warped = F.grid_sample(x, gridbw).squeeze_(1)
timages += [x_warped]

for x in timages:
    r = 1
    for c in vmf.grid_2d_points_to_indices(c_src + offsets, x.shape[-2:])[0]:
        x[..., c[0] - r: c[0] + r, c[1] - r:c[1] + r] = 1.

images += [to_pil_image(x.squeeze()) for x in timages]

fig, axs = plt.subplots(len(images))
for i, im in enumerate(images):
    axs[i].imshow(im)
plt.show()
