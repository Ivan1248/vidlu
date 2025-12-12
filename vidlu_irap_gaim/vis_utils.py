"""
Visualization utilities shared by `vidlu_irap_gaim` tools.

This module exists to avoid duplication between:
- `vidlu_irap_gaim/tools/dataset_viewer.py` (interactive browsing)
- `vidlu_irap_gaim/tools/inference_visualization.py` (offline inference + PNG writing)

It is deliberately dependency-light:
- Core functions use numpy/torch/PIL.
- OpenCV is imported only inside the functions that need resizing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AttributeMetadataDecoder:
    """
    Decodes per-attribute class indices into human-readable strings using IRAP metadata.

    Supports optional attribute value mapping (same JSON format used by `make_bih_data`).
    """

    attribute_names: list[str]
    attr_to_class_idx_to_value: dict[str, dict[int, str]]
    attr_to_values: dict[str, list[str]]

    @staticmethod
    def load(
        *,
        metadata_dir: str | Path,
        attribute_value_mapping_path: str | Path | None = None,
        expected_attribute_names: Sequence[str] | None = None,
    ) -> "AttributeMetadataDecoder":
        metadata_dir = Path(metadata_dir)
        with open(metadata_dir / "attribute_metadata.json", "r") as f:
            attr_meta = json.load(f)

        idx_to_attribute = {v: k for k, v in attr_meta["attribute_to_idx"].items()}
        ordered_attrs = [idx_to_attribute[i] for i in range(len(idx_to_attribute))]

        if attribute_value_mapping_path is not None:
            with open(attribute_value_mapping_path, "r") as f:
                attribute_to_value_to_new_value = json.load(f)
            # Match dataset behavior: filter attribute order to only mapped attributes
            ordered_attrs = [a for a in ordered_attrs if a in attribute_to_value_to_new_value]
        else:
            attribute_value_to_irap = attr_meta["attribute_value_to_irap_number"]
            attribute_to_value_to_new_value = {
                attr: {v: v for v in attribute_value_to_irap[attr].keys()} for attr in ordered_attrs
            }

        if expected_attribute_names is not None and list(expected_attribute_names) != list(ordered_attrs):
            raise ValueError(
                "Attribute order mismatch between dataset and metadata-derived mapping.\n"
                f"- dataset.info.attribute_names[:5]={list(expected_attribute_names)[:5]}\n"
                f"- metadata-derived ordered_attrs[:5]={ordered_attrs[:5]}\n"
                "Pass the same `attribute_value_mapping_path` you used to build the dataset."
            )

        attr_to_class_idx_to_value: dict[str, dict[int, str]] = {}
        attr_to_values: dict[str, list[str]] = {}
        for attr in ordered_attrs:
            values = list(attribute_to_value_to_new_value[attr].values())
            attr_to_values[attr] = values
            attr_to_class_idx_to_value[attr] = {i: v for i, v in enumerate(values)}

        return AttributeMetadataDecoder(
            attribute_names=list(ordered_attrs),
            attr_to_class_idx_to_value=attr_to_class_idx_to_value,
            attr_to_values=attr_to_values,
        )

    def value_str(self, *, attr: str, class_idx: int) -> str:
        return self.attr_to_class_idx_to_value.get(attr, {}).get(class_idx, str(class_idx))

    def to_text(self, *, attr: str, class_idx: int) -> str:
        return f"{attr}: {self.value_str(attr=attr, class_idx=class_idx)} ({class_idx})"

    def decode_label_tensor(self, labels: torch.Tensor) -> dict[str, tuple[str, int]]:
        """Decode a multi-attribute target tensor shaped (A,)."""
        res: dict[str, tuple[str, int]] = {}
        if labels is None:
            return res
        if labels.ndim != 1:
            return res
        if len(labels) != len(self.attribute_names):
            return res
        for i, attr in enumerate(self.attribute_names):
            class_idx = int(labels[i])
            res[attr] = (f"{self.value_str(attr=attr, class_idx=class_idx)} ({class_idx})", class_idx)
        return res


def tensor_image_to_uint8_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a (C, H, W) or (S, C, H, W) float tensor to (H, W, C) or (S, H, W, C) uint8 numpy array.
    Clips to [0, 1] and scales to [0, 255].
    """
    t = tensor.detach().cpu()

    if t.ndim == 4:
        # (S, C, H, W) -> (S, H, W, C)
        t = t.permute(0, 2, 3, 1)
    elif t.ndim == 3:
        # (C, H, W) -> (H, W, C)
        t = t.permute(1, 2, 0)

    t_np = t.numpy()
    t_np = np.clip(t_np, 0, 1)
    return (t_np * 255).astype(np.uint8)


def create_composite_view_strip(images: np.ndarray) -> np.ndarray | None:
    """
    Creates a composite image with the first image as 'main' and the rest as 'context' below it.
    Resizes the context strip to match the width of the main image.

    Input:
      - images: (S, H, W, C) uint8
    """
    if len(images) == 0:
        return None

    main_img = images[0]
    if len(images) == 1:
        return main_img

    # Lazy import: only dataset_viewer needs this.
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for create_composite_view_strip.") from e

    context_imgs = images[1:]

    # 1. Concatenate context images horizontally
    ctx_strip = np.concatenate(context_imgs, axis=1)

    # 2. Resize context strip to match main image width
    h_main, w_main = main_img.shape[:2]
    h_ctx, w_ctx = ctx_strip.shape[:2]

    if w_ctx > 0:
        scale = w_main / w_ctx
        new_w = w_main
        new_h = int(h_ctx * scale)

        # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        ctx_resized = cv2.resize(ctx_strip, (new_w, new_h), interpolation=interp)
    else:
        ctx_resized = ctx_strip

    # 3. Concatenate vertically
    return np.concatenate([main_img, ctx_resized], axis=0)


def rgb_seq_to_pil_images(rgb_seq: torch.Tensor) -> list[Image.Image]:
    """Convert (S, 3, H, W) float tensor in [0,1] to list of PIL images."""
    if rgb_seq.ndim != 4 or rgb_seq.shape[1] != 3:
        raise ValueError(f"Expected rgb_seq with shape (S, 3, H, W), got {tuple(rgb_seq.shape)}")
    t = rgb_seq.detach().cpu().clamp(0, 1)
    imgs: list[Image.Image] = []
    for i in range(t.shape[0]):
        arr = (t[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def make_grid_pil(images: list[Image.Image], *, out_w: int, out_h: int) -> Image.Image:
    """Pack images into a grid canvas of size (out_w, out_h) with aspect-preserving fits."""
    canvas = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    if not images:
        return canvas

    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = max(1, out_w // cols)
    cell_h = max(1, out_h // rows)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        img_aspect = img.width / max(1, img.height)
        cell_aspect = cell_w / max(1, cell_h)
        if img_aspect > cell_aspect:
            new_w = cell_w
            new_h = max(1, int(new_w / img_aspect))
        else:
            new_h = cell_h
            new_w = max(1, int(new_h * img_aspect))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x0 = c * cell_w + (cell_w - new_w) // 2
        y0 = r * cell_h + (cell_h - new_h) // 2
        canvas.paste(resized, (x0, y0))

    return canvas


def _wrap_text_lines(text: str, *, max_chars: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.rstrip("\n")
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars + 1)
            if cut <= 0:
                cut = max_chars
            lines.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        lines.append(s)
    return lines


def render_text_panel_pil(
    text: str,
    *,
    width: int,
    height: int,
    padding: int = 18,
    bg: tuple[int, int, int] = (0, 0, 0),
    fg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    approx_char_w = 7
    max_chars = max(10, (width - 2 * padding) // approx_char_w)
    lines = _wrap_text_lines(text, max_chars=max_chars)

    y = padding
    line_h = font.getbbox("Ag")[3] + 4
    for line in lines:
        if y + line_h > height - padding:
            break
        draw.text((padding, y), line, font=font, fill=fg)
        y += line_h
    return img


def make_inference_visualization_image(
    *,
    rgb_seq: torch.Tensor,
    text: str,
    out_size: tuple[int, int] = (1920, 1080),
    text_area_ratio: float = 0.35,
) -> Image.Image:
    out_w, out_h = out_size
    text_w = int(out_w * text_area_ratio)
    img_w = out_w - text_w

    frames = rgb_seq_to_pil_images(rgb_seq)
    grid = make_grid_pil(frames, out_w=img_w, out_h=out_h)
    panel = render_text_panel_pil(text, width=text_w, height=out_h)

    combined = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    combined.paste(grid, (0, 0))
    combined.paste(panel, (img_w, 0))
    return combined


def save_inference_visualization(
    *,
    rgb_seq: torch.Tensor,
    text: str,
    segment_id: str,
    output_dir: str | Path,
    out_size: tuple[int, int] = (1920, 1080),
    text_area_ratio: float = 0.35,
) -> Path:
    img = make_inference_visualization_image(
        rgb_seq=rgb_seq, text=text, out_size=out_size, text_area_ratio=text_area_ratio
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{segment_id}_prediction.png"
    img.save(out_path)
    return out_path


