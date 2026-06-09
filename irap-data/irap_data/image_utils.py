"""Image loading and preprocessing helpers shared by data-loading code."""

import math

import cv2
import numpy as np
import torch


def load_image_cv2(path: str) -> np.ndarray:
    """Load an image as an HWC uint8 RGB ndarray."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def rgb_to_chw_tensor(
    img_hwc: np.ndarray,
    target_wh: tuple[int, int],
) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float32 tensor in [0, 1].

    Resizes-to-cover (Lanczos), then center-crops to target_wh. The resize
    fires only when some source dimension is below the target. When the
    source matches target_wh exactly, both steps leave pixels unchanged.
    """
    from PIL import Image

    pil = Image.fromarray(img_hwc)
    w, h = pil.size
    target_w, target_h = target_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(math.ceil(w * scale)))
    new_h = max(target_h, int(math.ceil(h * scale)))
    if (new_w, new_h) != (w, h):
        pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    pil = pil.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(pil).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(arr)
