import numpy as np
import torch

from .transformers import image_transformer, is_numpy_image, is_pil_image, is_torch_image


def prepare_input_label(y):
    if isinstance(y, np.ndarray) and y.dtype == np.int8:  # TODO: remove when PyTorch is fixed
        y = torch.tensor(y.astype(np.uint8), dtype=torch.int8)
    if isinstance(y, torch.Tensor) and 'int' in str(y.dtype):
        return y.long()
    elif isinstance(y, (int, np.integer)):
        return torch.tensor(y, dtype=torch.int64)
    return y  # DataLoader will convert it to Torch
