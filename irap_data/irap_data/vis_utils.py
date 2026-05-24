"""Visualization helpers used by the dataset viewer.

Subset of utilities sufficient for `dataset_viewer.py`. Kept minimal so the
package's runtime dependencies stay light (numpy/torch; OpenCV only when
multiple context frames need to be composited).
"""

from dataclasses import dataclass, field

import numpy as np
import torch


# Palette with ~25 distinct colors for good contrast on dark/light backgrounds.
# Index 0 is gray (often represents 'None' or 'Unknown').
_CLASS_COLOR_PALETTE = (
    "#808080",  # 0: Gray (None/Unknown)
    "#E6194B",  # 1: Red
    "#3CB44B",  # 2: Green
    "#FFE119",  # 3: Yellow
    "#4363D8",  # 4: Blue
    "#F58231",  # 5: Orange
    "#911EB4",  # 6: Purple
    "#42D4F4",  # 7: Cyan
    "#F032E6",  # 8: Magenta
    "#BFEF45",  # 9: Lime
    "#FABED4",  # 10: Pink
    "#469990",  # 11: Teal
    "#DCBEFF",  # 12: Lavender
    "#9A6324",  # 13: Brown
    "#FFFAC8",  # 14: Beige
    "#800000",  # 15: Maroon
    "#AAFFC3",  # 16: Mint
    "#808000",  # 17: Olive
    "#FFD8B1",  # 18: Apricot
    "#000075",  # 19: Navy
    "#A9A9A9",  # 20: Dark Gray
    "#E6BEFF",  # 21: Light Purple
    "#AA6E28",  # 22: Tan
    "#00FA9A",  # 23: Spring Green
    "#FF6347",  # 24: Tomato
)


def get_index_color(idx: int) -> str:
    """Hex color code for a class index. Index 0 returns gray."""
    if idx < 0:
        return _CLASS_COLOR_PALETTE[0]
    if idx < len(_CLASS_COLOR_PALETTE):
        return _CLASS_COLOR_PALETTE[idx]
    return _CLASS_COLOR_PALETTE[1 + (idx - 1) % (len(_CLASS_COLOR_PALETTE) - 1)]


@dataclass
class AttributeMetadataDecoder:
    """Decodes per-attribute class indices into human-readable strings."""

    attr_to_value_to_class_idx: dict[str, dict[str, int]]
    attr_to_class_idx_to_value: dict[str, dict[int, str]] = field(init=False)
    ignore_class_idx: int | None = -1

    def __post_init__(self):
        self.attr_to_class_idx_to_value = {attr: {i: v for v, i in self.attr_to_value_to_class_idx[attr].items()}
                                           for attr in self.attr_to_value_to_class_idx.keys()}

    def value_str(self, *, attr: str, class_idx: int) -> str:
        if self.ignore_class_idx is not None and class_idx == self.ignore_class_idx:
            return "(unlabeled)"
        return self.attr_to_class_idx_to_value.get(attr, {}).get(class_idx, "(unknown label)")

    def to_text(self, *, attr: str, class_idx: int) -> str:
        return f"{attr}: {self.value_str(attr=attr, class_idx=class_idx)} ({class_idx})"

    def decode_label_tensor(self, labels: torch.Tensor) -> dict[str, tuple[str, int]]:
        res: dict[str, tuple[str, int]] = {}
        if labels is None or labels.ndim != 1 or len(labels) != len(self.attr_to_class_idx_to_value):
            return res
        for i, attr in enumerate(self.attr_to_class_idx_to_value.keys()):
            class_idx = int(labels[i])
            res[attr] = (f"{self.value_str(attr=attr, class_idx=class_idx)} ({class_idx})", class_idx)
        return res


def tensor_image_to_uint8_np(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) or (S, C, H, W) float tensor in [0, 1] -> uint8 HWC / SHWC array."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)
    elif t.ndim == 3:
        t = t.permute(1, 2, 0)
    return (np.clip(t.numpy(), 0, 1) * 255).astype(np.uint8)


def create_composite_view_strip(images: np.ndarray) -> np.ndarray | None:
    """Stack the first frame as 'main' on top, the rest as a context strip below."""
    if len(images) == 0:
        return None

    main_img = images[0]
    if len(images) == 1:
        return main_img

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for create_composite_view_strip.") from e

    context_imgs = images[1:]
    ctx_strip = np.concatenate(context_imgs, axis=1)

    h_main, w_main = main_img.shape[:2]
    h_ctx, w_ctx = ctx_strip.shape[:2]

    if w_ctx > 0:
        scale = w_main / w_ctx
        new_w = w_main
        new_h = int(h_ctx * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        ctx_resized = cv2.resize(ctx_strip, (new_w, new_h), interpolation=interp)
    else:
        ctx_resized = ctx_strip

    return np.concatenate([main_img, ctx_resized], axis=0)
