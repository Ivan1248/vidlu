import numpy as np
import torch

from vidlu.utils.func import compose
import vidlu.transforms.image as vti


def prepare_label(y):
    if isinstance(y, np.ndarray):
        if 'float' in str(y.dtype):
            return torch.tensor(y)
        else:
            return torch.tensor(y, dtype=torch.int64)
    elif isinstance(y, (int, np.integer)):
        return torch.tensor(y, dtype=torch.int64)
    return y  # DataLoader will convert it to Torch


def prepare_input_image(x):
    return compose(vti.to_torch, vti.hwc_to_chw, lambda x: x.to(dtype=torch.float), vti.Div(255))(x)
