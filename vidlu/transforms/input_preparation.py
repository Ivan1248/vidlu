import numpy as np
import torch

from vidlu.utils.func import compose
from . import image as imt


def prepare_label(y):
    """ Turns the label into torch.int64"""
    if isinstance(y, np.ndarray):
        if 'float' in str(y.dtype):
            return torch.tensor(y)
        else:
            return torch.tensor(y, dtype=torch.int64)
    elif isinstance(y, (int, np.integer)):
        return torch.tensor(y, dtype=torch.int64)
    return y  # DataLoader will convert it to Torch


def prepare_input_image(x):
    return compose(imt.to_torch, imt.hwc_to_chw, imt.To(dtype=torch.float), imt.Div(255))(x)
