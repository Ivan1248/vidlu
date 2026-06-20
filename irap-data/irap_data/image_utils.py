"""Image loading and preprocessing helpers shared by data-loading code."""

import cv2
import numpy as np
import torch


def load_image_cv2(path: str) -> np.ndarray:
    """Load an image as an HWC uint8 RGB ndarray."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def center_crop(img_hwc: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Center-crop an HWC ndarray to target_wh without interpolation.

    Raises ValueError if the source is smaller than the target in any dimension.
    """
    h, w = img_hwc.shape[:2]
    target_w, target_h = target_wh
    if w < target_w or h < target_h:
        raise ValueError(
            f"Source image ({w}×{h}) is smaller than target ({target_w}×{target_h}). "
            f"Call resize_to_cover first if the source size is not guaranteed."
        )
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img_hwc[top:top + target_h, left:left + target_w]


def resize_to_cover(img_hwc: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Resize an HWC ndarray so both dimensions meet or exceed target_wh.

    Preserves aspect ratio (resize-to-cover). Uses Lanczos resampling via PIL.
    Call center_crop afterwards to get the exact target size.
    """
    import math
    from PIL import Image

    h, w = img_hwc.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    target_w, target_h = target_wh
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, int(math.ceil(w * scale)))
    new_h = max(target_h, int(math.ceil(h * scale)))
    if (new_w, new_h) == (w, h):
        return img_hwc
    pil = Image.fromarray(img_hwc)
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(pil)


def hwc_to_chw_float_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 ndarray → CHW float32 tensor in [0, 1]."""
    return torch.from_numpy(img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)
