"""
Visualization utilities shared by `vidlu_irap_gaim` tools.

This module exists to avoid duplication between:
- `vidlu_irap_gaim/tools/dataset_viewer.py` (interactive browsing)
- `vidlu_irap_gaim/tools/inference_visualization.py` (offline inference + PNG writing)

It is deliberately dependency-light:
- Core functions use numpy/torch/PIL.
- OpenCV is imported only inside the functions that need resizing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# --- Color palette for class indices ---

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
    """
    Returns a hex color code for a class index.

    Index 0 returns gray (often 'None'). Other indices cycle through the palette.
    """
    if idx < 0:
        return _CLASS_COLOR_PALETTE[0]
    if idx < len(_CLASS_COLOR_PALETTE):
        return _CLASS_COLOR_PALETTE[idx]
    # Wrap around for indices beyond palette size (skip index 0 for non-zero classes)
    return _CLASS_COLOR_PALETTE[1 + (idx - 1) % (len(_CLASS_COLOR_PALETTE) - 1)]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


@dataclass
class AttributeMetadataDecoder:
    """
    Decodes per-attribute class indices into human-readable strings using IRAP metadata.

    Supports optional attribute value mapping (same JSON format used by `make_bih_data`).
    """
    attr_to_value_to_class_idx: dict[str, dict[str, int]]
    attr_to_class_idx_to_value: dict[str, dict[int, str]] = field(init=False)

    def __post_init__(self):
        self.attr_to_class_idx_to_value = {attr: {i: v for v, i in self.attr_to_value_to_class_idx[attr].items()}
                                           for attr in self.attr_to_value_to_class_idx.keys()}

    def value_str(self, *, attr: str, class_idx: int) -> str:
        return self.attr_to_class_idx_to_value.get(attr, {}).get(class_idx, "(?)")

    def to_text(self, *, attr: str, class_idx: int) -> str:
        return f"{attr}: {self.value_str(attr=attr, class_idx=class_idx)} ({class_idx})"

    def decode_label_tensor(self, labels: torch.Tensor) -> dict[str, tuple[str, int]]:
        """Decode a multi-attribute target tensor shaped (A,)."""
        res: dict[str, tuple[str, int]] = {}
        if labels is None:
            return res
        if labels.ndim != 1:
            return res
        if len(labels) != len(self.attr_to_class_idx_to_value):
            return res
        for i, attr in enumerate(self.attr_to_class_idx_to_value.keys()):
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


def format_prediction_line(
    *,
    attr: str,
    pred_value: str,
    pred_idx: int,
    prob: float | None,
    gt_value: str | None = None,
    gt_idx: int | None = None,
    gt_prob: float | None = None,
) -> str:
    """
    Format a single attribute prediction line for visualization.

    Intended output examples:
      - "Curvature: Straight (0) [98.2%]"
      - "Lane width: Wide (2) [85.7%] ✓"
      - "Road condition: Good (1) [67.3%] ✗ GT: Poor (2) [12.1%]"
    """
    prob_str = "" if prob is None else f" [{100.0 * float(prob):.1f}%]"
    base = f"{attr}: {pred_value} ({pred_idx}){prob_str}"
    if gt_idx is None:
        return base
    if int(gt_idx) == int(pred_idx):
        return f"{base} ✓"
    gt_str = gt_value if gt_value is not None else str(int(gt_idx))
    gt_prob_str = "" if gt_prob is None else f" [{100.0 * float(gt_prob):.1f}%]"
    return f"{base} ✗ GT: {gt_str} ({int(gt_idx)}){gt_prob_str}"


def render_prediction_panel_pil(
    lines: Sequence[str] | Sequence[tuple[str, tuple[int, int, int]]],
    *,
    width: int,
    height: int,
    padding: int = 18,
    bg: tuple[int, int, int] = (0, 0, 0),
    fg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Render a text panel with optional per-line colors.

    Args:
        lines: Either a list of strings (all drawn using fg), or a list of (text, rgb_color).
    """
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Normalize to list[(text, color)]
    colored_lines: list[tuple[str, tuple[int, int, int]]] = []
    for x in lines:
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
            colored_lines.append((x[0], x[1]))
        else:
            colored_lines.append((str(x), fg))

    approx_char_w = 7
    max_chars = max(10, (width - 2 * padding) // approx_char_w)
    y = padding
    line_h = font.getbbox("Ag")[3] + 4

    for raw_text, color in colored_lines:
        for line in _wrap_text_lines(raw_text, max_chars=max_chars):
            if y + line_h > height - padding:
                return img
            draw.text((padding, y), line, font=font, fill=color)
            y += line_h
    return img


@dataclass
class PredictionRow:
    """Data for a single attribute prediction to be visualized."""

    attr: str
    pred_value: str
    pred_idx: int
    pred_prob: float
    gt_value: str | None = None
    gt_idx: int | None = None
    gt_prob: float | None = None

    @property
    def is_correct(self) -> bool:
        return self.gt_idx is None or self.gt_idx == self.pred_idx


def render_prediction_panel_rich(
    rows: Sequence[PredictionRow],
    *,
    width: int,
    height: int,
    padding: int = 18,
    bar_height: int = 12,
    bar_width: int = 60,
    row_spacing: int = 6,
    bg: tuple[int, int, int] = (0, 0, 0),
    fg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Render a prediction panel with colored class values and probability comparison bars.

    Layout per attribute (single line):
      [Attribute]: [Pred value] [pred_bar XX%] ✓/✗ [GT value] [gt_bar YY%]

    Args:
        rows: Sequence of PredictionRow objects with prediction data.
        width, height: Output image dimensions.
        padding: Padding around content.
        bar_height: Height of probability bars in pixels.
        bar_width: Width of probability bars in pixels.
        row_spacing: Vertical spacing between attribute rows.
        bg: Background color (RGB).
        fg: Default text color (RGB).
    """
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def draw_bold_text(xy: tuple[int, int], s: str, *, fill: tuple[int, int, int]) -> None:
        # PIL default bitmap font has no bold variant; emulate by drawing twice with 1px offset.
        x, y = xy
        draw.text((x, y), s, font=font, fill=fill)
        draw.text((x + 1, y), s, font=font, fill=fill)

    def text_width(s: str) -> int:
        """Get actual rendered width of text."""
        bbox = font.getbbox(s)
        return bbox[2] - bbox[0]

    # Measure text height
    line_h = font.getbbox("Ag")[3] * 1.1

    y = padding
    x_base = padding
    gap = 6  # Small gap between elements

    bar_bg = (60, 60, 60)

    for row in rows:
        if y + line_h + row_spacing > height - padding:
            break

        pred_color = hex_to_rgb(get_index_color(row.pred_idx))
        x_cursor = x_base

        # Draw attribute name
        attr_text = f"{row.attr}:"
        draw_bold_text((x_cursor, y), attr_text, fill=fg)
        # Single flowing column: place the class immediately after the attribute label.
        x_cursor += text_width(attr_text) + gap

        # Draw predicted value (colored)
        pred_text = f"{row.pred_value} ({row.pred_idx})"
        draw.text((x_cursor, y), pred_text, font=font, fill=pred_color)
        x_cursor += text_width(pred_text) + gap

        # Draw probability bar for prediction
        x_bar = x_cursor
        draw.rectangle(
            [x_bar, y, x_bar + bar_width, y + bar_height],
            fill=bar_bg,
            outline=None,
        )
        fill_w = int(bar_width * row.pred_prob)
        if fill_w > 0:
            draw.rectangle(
                [x_bar, y, x_bar + fill_w, y + bar_height],
                fill=pred_color,
                outline=None,
            )
        # Probability text (black) centered inside the bar
        prob_text = f"{100 * row.pred_prob:.0f}%"
        prob_w = text_width(prob_text)
        prob_x = x_bar + max(0, (bar_width - prob_w) // 2)
        prob_y = y + max(0, (bar_height - line_h) // 2)
        draw.text((prob_x, prob_y), prob_text, font=font, fill=(0, 0, 0))
        x_cursor = x_bar + bar_width + gap

        # Check/cross indicator
        if row.gt_idx is not None:
            if row.is_correct:
                draw.text((x_cursor, y), "✓", font=font, fill=(100, 255, 100))
            else:
                draw.text((x_cursor, y), "✗", font=font, fill=(255, 100, 100))
            x_cursor += 12

            # If incorrect, draw GT on the same line
            if not row.is_correct and row.gt_prob is not None:
                gt_color = hex_to_rgb(get_index_color(row.gt_idx))
                gt_value_str = row.gt_value if row.gt_value else f"({row.gt_idx})"

                # GT label
                gt_label = f"GT: {gt_value_str} ({row.gt_idx})"
                draw.text((x_cursor, y), gt_label, font=font, fill=gt_color)
                x_cursor += text_width(gt_label) + gap

                # GT probability bar
                x_gt_bar = x_cursor
                draw.rectangle(
                    [x_gt_bar, y, x_gt_bar + bar_width, y + bar_height],
                    fill=bar_bg,
                    outline=None,
                )
                gt_fill_w = int(bar_width * row.gt_prob)
                if gt_fill_w > 0:
                    draw.rectangle(
                        [x_gt_bar, y, x_gt_bar + gt_fill_w, y + bar_height],
                        fill=gt_color,
                        outline=None,
                    )
                # GT probability text (black) centered inside the bar
                gt_prob_text = f"{100 * row.gt_prob:.0f}%"
                gt_prob_w = text_width(gt_prob_text)
                gt_prob_x = x_gt_bar + max(0, (bar_width - gt_prob_w) // 2)
                gt_prob_y = y + max(0, (bar_height - line_h) // 2)
                draw.text((gt_prob_x, gt_prob_y), gt_prob_text, font=font, fill=(0, 0, 0))
                x_cursor = x_gt_bar + bar_width + gap

        y += line_h + row_spacing

    return img


def composite_to_fitted_pil(
    rgb_seq: torch.Tensor,
    *,
    out_w: int,
    out_h: int,
    align: str = "left",
) -> Image.Image:
    """
    Convert an RGB sequence tensor to a composite image fitted to (out_w, out_h).

    Args:
        rgb_seq: (S, 3, H, W) float tensor in [0, 1]
        out_w, out_h: Target canvas dimensions
        align: Horizontal alignment - "left", "center", or "right"

    Returns:
        PIL Image with composite fitted and aligned on black canvas.
    """
    import cv2

    imgs_np = tensor_image_to_uint8_np(rgb_seq)
    composite = create_composite_view_strip(imgs_np)

    if composite is None:
        return Image.new("RGB", (out_w, out_h), color=(0, 0, 0))

    # Resize to fit while preserving aspect ratio
    h, w = composite.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(composite, (new_w, new_h), interpolation=interp)

    # Align on canvas
    canvas = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    pil_img = Image.fromarray(resized)
    if align == "left":
        x_offset = 0
    elif align == "right":
        x_offset = out_w - new_w
    else:  # center
        x_offset = (out_w - new_w) // 2
    y_offset = (out_h - new_h) // 2
    canvas.paste(pil_img, (x_offset, y_offset))
    return canvas


def composite_to_fitted_pil_layout(
    rgb_seq: torch.Tensor,
    *,
    out_w: int,
    out_h: int,
) -> tuple[Image.Image, int, int]:
    """
    Like `composite_to_fitted_pil`, but returns the *resized image only* (no padding canvas),
    plus its (new_w, y_offset) when placed on an (out_w, out_h) canvas at x=0.

    This is useful when you want to place the text panel right after the rendered image content,
    instead of after the (potentially wider) image panel.
    """
    import cv2

    imgs_np = tensor_image_to_uint8_np(rgb_seq)
    composite = create_composite_view_strip(imgs_np)
    if composite is None:
        return Image.new("RGB", (1, 1), color=(0, 0, 0)), 1, 0

    h, w = composite.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(composite, (new_w, new_h), interpolation=interp)
    pil_img = Image.fromarray(resized)
    y_offset = max(0, (out_h - new_h) // 2)
    return pil_img, new_w, y_offset


def make_inference_visualization_image(
    *,
    rgb_seq: torch.Tensor,
    text: str,
    out_size: tuple[int, int] = (1920, 1080),
    text_area_ratio: float = 0.7,
    gap: int = 0,
) -> Image.Image:
    """
    Create a visualization image with composite frames on left and text panel on right.

    Args:
        rgb_seq: (S, 3, H, W) float tensor in [0, 1]
        text: Text to render in the panel
        out_size: (width, height) of output image
        text_area_ratio: Fraction of width for text panel
        gap: Pixel gap between image and text panel
    """
    out_w, out_h = out_size
    min_text_w = int(out_w * text_area_ratio)
    img_max_w = max(1, out_w - min_text_w - gap)

    # Fit image into the max image width, but place text right after the *actual* rendered image width.
    img_resized, img_used_w, img_y = composite_to_fitted_pil_layout(rgb_seq, out_w=img_max_w, out_h=out_h)
    x_text = min(out_w - 1, img_used_w + gap)
    text_w = max(1, out_w - x_text)
    text_panel = render_text_panel_pil(text, width=text_w, height=out_h)

    combined = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    combined.paste(img_resized, (0, img_y))
    combined.paste(text_panel, (x_text, 0))
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
