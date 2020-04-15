import torch
import torch.nn.functional as F

# noinspection PyUnresolvedReferences
import _context
import dirs
from vidlu import modules, factories
from vidlu.utils.presentation.visualization import Viewer
from vidlu.transforms import image

data = factories.get_prepared_data_for_trainer("Cityscapes(downsampling=4){train}", dirs.DATASETS,
                                               dirs.CACHE).train


def filter_(x, sigma):
    return modules.components.GaussianFilter2D(sigma=sigma)(x)


def high_pass(x, sigma=2):
    return x - filter_(x, sigma)


def local_mean(x):
    return filter_(x, 4)


def local_variance(x):
    return local_mean(x ** 2) - local_mean(x) ** 2


def hf_local_variance(x):
    return local_variance(high_pass(x))


def pyramid(x, factors=(1, 2, 4, 8, 16, 32)):
    size = x.shape[-2:]
    pyr = [F.interpolate(x, scale_factor=1 / s, mode='bilinear', align_corners=False)
           for s in factors]
    pyr = map(local_variance, pyr)
    return [F.interpolate(x, size=size, mode='bilinear', align_corners=False) for x in pyr]


def combined_scale(x):
    pyr = pyramid(x)
    return torch.stack(pyr, 0).mean(0)[0]


def torch_to_numpy(x):
    return image.chw_to_hwc(x).cpu().numpy()


@torch.no_grad()
def apply_single(fun, x):
    return fun(x.unsqueeze(0)).squeeze(0)


Viewer().display(data,
                 lambda r: [torch_to_numpy(r.x), torch_to_numpy(r.x) / torch_to_numpy(
                     12 * apply_single(combined_scale, r.x.cuda()) ** 0.5)])
