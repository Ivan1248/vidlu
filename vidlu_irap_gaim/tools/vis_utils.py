"""
Visualization utilities of the `vidlu_irap_gaim` inference tools.

A visualization shows the frames of a segment on the left and a text panel right after them.
The class colors and the frame composite are those of the `irap_data` dataset viewer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

from irap_data.tools.viewer.vis_utils import (
    create_composite_view_strip,
    get_index_color,
    tensor_image_to_uint8_np,
)


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

    def draw_probability_bar(x: int, y: int, prob: float, color: str) -> int:
        """Draws a bar filled to `prob`, with the percentage centered in it, and returns the x
        position after it."""
        draw.rectangle([x, y, x + bar_width, y + bar_height], fill=bar_bg)
        fill_w = int(bar_width * prob)
        if fill_w > 0:
            draw.rectangle([x, y, x + fill_w, y + bar_height], fill=color)
        prob_text = f"{100 * prob:.0f}%"
        prob_x = x + max(0, (bar_width - text_width(prob_text)) // 2)
        prob_y = y + max(0, (bar_height - line_h) // 2)
        draw.text((prob_x, prob_y), prob_text, font=font, fill=(0, 0, 0))
        return x + bar_width + gap

    for row in rows:
        if y + line_h + row_spacing > height - padding:
            break

        pred_color = get_index_color(row.pred_idx)
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

        x_cursor = draw_probability_bar(x_cursor, y, row.pred_prob, pred_color)

        # Check/cross indicator
        if row.gt_idx is not None:
            if row.is_correct:
                draw.text((x_cursor, y), "✓", font=font, fill=(100, 255, 100))
            else:
                draw.text((x_cursor, y), "✗", font=font, fill=(255, 100, 100))
            x_cursor += 12

            # If incorrect, draw GT on the same line
            if not row.is_correct and row.gt_prob is not None:
                gt_color = get_index_color(row.gt_idx)
                gt_value_str = row.gt_value if row.gt_value else f"({row.gt_idx})"

                # GT label
                gt_label = f"GT: {gt_value_str} ({row.gt_idx})"
                draw.text((x_cursor, y), gt_label, font=font, fill=gt_color)
                x_cursor += text_width(gt_label) + gap

                x_cursor = draw_probability_bar(x_cursor, y, row.gt_prob, gt_color)

        y += line_h + row_spacing

    return img


def _fit_frames(rgb_seq: torch.Tensor, *, max_w: int, max_h: int) -> Image.Image:
    """Composites the frames (see `create_composite_view_strip`) and resizes the composite to
    fit (max_w, max_h), preserving its aspect ratio.

    Args:
        rgb_seq: (S, 3, H, W) float tensor in [0, 1].
    """
    import cv2

    composite = create_composite_view_strip(tensor_image_to_uint8_np(rgb_seq))
    if composite is None:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))

    h, w = composite.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return Image.fromarray(cv2.resize(composite, (new_w, new_h), interpolation=interp))


def make_visualization_image(
    rgb_seq: torch.Tensor,
    render_panel: Callable[[int, int], Image.Image],
    *,
    out_size: tuple[int, int] = (1920, 1080),
    text_area_ratio: float = 0.35,
    gap: int = 0,
) -> Image.Image:
    """Places the frames of a segment on the left and a panel right after them.

    The frames are fitted into the width that remains after `text_area_ratio` of the width is
    reserved for the panel. The panel then takes all the width that the fitted frames leave.

    Args:
        rgb_seq: (S, 3, H, W) float tensor in [0, 1].
        render_panel: Renders the panel with a given (width, height).
        out_size: (width, height) of the image.
        text_area_ratio: The fraction of the width reserved for the panel.
        gap: Pixel gap between the frames and the panel.
    """
    out_w, out_h = out_size
    frames = _fit_frames(rgb_seq, max_w=max(1, out_w - int(out_w * text_area_ratio) - gap),
                         max_h=out_h)
    x_panel = min(out_w - 1, frames.width + gap)
    image = Image.new("RGB", (out_w, out_h), color=(0, 0, 0))
    image.paste(frames, (0, max(0, (out_h - frames.height) // 2)))
    image.paste(render_panel(max(1, out_w - x_panel), out_h), (x_panel, 0))
    return image


def save_inference_visualization(
    *,
    rgb_seq: torch.Tensor,
    text: str,
    segment_id: str,
    output_dir: str | Path,
    out_size: tuple[int, int] = (1920, 1080),
    text_area_ratio: float = 0.35,
) -> Path:
    img = make_visualization_image(
        rgb_seq, lambda w, h: render_text_panel_pil(text, width=w, height=h),
        out_size=out_size, text_area_ratio=text_area_ratio)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{segment_id}_prediction.png"
    img.save(out_path)
    return out_path
