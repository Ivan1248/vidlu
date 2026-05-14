"""
Image format conversion utilities for VLM predictors.

Converts between PIL Images, NumPy arrays, and PyTorch tensors.
"""

import numpy as np
import torch
from PIL import Image


def to_pil_image(
    image: Image.Image | np.ndarray | torch.Tensor,
) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        # Assume (C, H, W) or (H, W, C) format
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4):  # (C, H, W)
                image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
        elif image.ndim == 2:
            image = image.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")

    if isinstance(image, np.ndarray):
        # Handle float [0, 1] -> uint8 [0, 255]
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return Image.fromarray(image)

    raise TypeError(f"Unsupported image type: {type(image)}")
