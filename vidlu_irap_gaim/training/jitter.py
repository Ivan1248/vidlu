import torch
from torchvision.transforms import transforms as T_trans


# Jitter presets (single source of truth)
JITTER_STANDARD = dict(brightness=0.6, contrast=0.3, saturation=0.2, hue=0.02)
JITTER_STRONG = dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)


def make_sequence_color_jitter(
    *,
    brightness: float = None,
    contrast: float = None,
    saturation: float = None,
    hue: float = None,
    preset: dict = None,
):
    """
    Color jitter for sequence tensors stored in records.

    Expects rgb frames in [0, 1] (normalization handled by input adapter).
    Applies ColorJitter frame-by-frame.

    Args:
        brightness, contrast, saturation, hue: Individual parameters (override preset).
        preset: A dict with jitter parameters (e.g. JITTER_STANDARD, JITTER_STRONG).
                If None, uses JITTER_STANDARD.
    """
    if preset is None:
        preset = JITTER_STANDARD

    # Allow individual params to override preset
    params = dict(preset)
    if brightness is not None:
        params["brightness"] = brightness
    if contrast is not None:
        params["contrast"] = contrast
    if saturation is not None:
        params["saturation"] = saturation
    if hue is not None:
        params["hue"] = hue

    color_jitter = T_trans.ColorJitter(**params)

    def _apply(record):
        if "rgb" not in record.keys():
            return record

        rgb = record["rgb"]
        if rgb.ndim not in (4, 5):
            return record

        if rgb.shape[0] == 0:
            return record

        if rgb.ndim == 4:
            jittered_frames = [color_jitter(frame) for frame in rgb]
            jittered = torch.stack(jittered_frames, dim=0)
        else:  # ndim == 5: (B, T, C, H, W)
            jittered_batches = []
            for b_idx in range(rgb.shape[0]):
                video = rgb[b_idx]
                jittered_frames = [color_jitter(frame) for frame in video]
                jittered_batches.append(torch.stack(jittered_frames, dim=0))
            jittered = torch.stack(jittered_batches, dim=0)

        return type(record)(record, rgb=jittered)

    return _apply
