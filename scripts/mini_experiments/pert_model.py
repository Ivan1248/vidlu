import torch
import torchvision.transforms.functional as tvtf
import matplotlib.pyplot as plt

from vidlu.configs.robustness import ph3_attack, phtps20_attack
from vidlu.data.datasets import HBlobs
from vidlu.data.types import Image, SegMap
from torchvision.utils import make_grid

# attack = ph3_attack()
attack = phtps20_attack()


def sample_pmodel(*args):
    return attack(lambda x: x, *args)


ds = HBlobs().map(lambda r: type(r)(image=tvtf.to_tensor(r.image)))

img = ds[0].image.unsqueeze(0)  # NCHW
mask = torch.ones((img.shape[0], *img.shape[2:]))
segmap = torch.ones(img.shape)

pmodel = sample_pmodel(img)

img_p, segmap_p, mask_p = pmodel(img, segmap, mask)

mask_to_image = lambda m: m.unsqueeze(1).expand(m.shape[0], 3, *m.shape[1:])

img_grid = make_grid(torch.cat(
    [img, segmap, mask_to_image(mask), img_p, segmap_p, mask_to_image(mask_p)],
    dim=0), nrow=3)

plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
plt.show()
