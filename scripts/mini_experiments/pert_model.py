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

img = ds[0].image.unsqueeze(0)
img = Image(img)  # ovako će vjerojatno trebati u budućnosti radi fleksibilnosti

pmodel = sample_pmodel(img)

img_p = pmodel(img)

img_grid = make_grid(torch.cat([img, img_p], dim=0))

plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
plt.show()
