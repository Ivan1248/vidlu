import numpy as np
import torch

from .transforms import image_transformer, is_numpy_image, is_pil_image, is_torch_image


def prepare_input_image(x, mean=None, std=None):
    if (mean is None) != (std is None):
        raise ValueError("Either both mean and std need to be provided or none of them.")
    if not (is_torch_image(x) or is_pil_image(x) or is_numpy_image(x)):
        raise TypeError("Unsupported input type or format.")
    tr = image_transformer(x).to_torch()
    if mean is not None:
        tr = tr.standardize(mean, std)
    return tr.transpose_to_chw().item


def prepare_input_label(x):
    if isinstance(x, np.ndarray) and x.dtype == np.int8:
        return x.astype(np.int16)
    return x  # DataLoader will convert it to Torch
