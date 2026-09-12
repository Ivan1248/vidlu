"""Hugging Face ViT backbones (DINOv2) and helpers shared by the ViT-style encoders."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .base import FrameEncoder, PixelStats

DINOV2_VARIANT_TO_HF_REPO = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}

# Enough to build the model and know its preprocessing, without pulling the duplicate
# PyTorch-pickle weights that most repos also ship.
_HF_MODEL_FILE_PATTERNS = ("config.json", "preprocessor_config.json", "*.safetensors",
                           "*.safetensors.index.json")


def patch_tokens_to_feature_map(
    patch_tokens: torch.Tensor,
    frame: torch.Tensor,
    patch_size: int | None,
) -> torch.Tensor:
    """Reshapes a `(B, N, D)` sequence of patch tokens into a `(B, D, grid_h, grid_w)`
    spatial feature map.

    The grid is inferred from the input `frame` size and `patch_size`; if that does
    not match the token count (e.g. extra register tokens), it falls back to the
    nearest square factorization.
    """
    B, patch_count, dim = patch_tokens.shape
    if patch_size is None:
        raise RuntimeError("patch_size is required to compute the spatial layout.")
    grid_h = frame.shape[-2] // patch_size
    grid_w = frame.shape[-1] // patch_size
    if grid_h * grid_w != patch_count:
        grid_h = int(round(patch_count**0.5))
        grid_w = patch_count // grid_h
        if grid_h * grid_w != patch_count:
            raise ValueError(f"Cannot reshape {patch_count} tokens into a grid.")
    return patch_tokens.transpose(1, 2).reshape(B, dim, grid_h, grid_w)


def download_hf_model(repo_id: str, params_dir: str | Path) -> Path:
    """Downloads Hugging Face model config, preprocessor config, and weights.

    Args:
        repo_id: Hugging Face model repository identifier.
        params_dir: Directory where downloaded model files are cached.

    Returns:
        Path to the local directory containing downloaded files.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, cache_dir=str(Path(params_dir).expanduser()),
                                  allow_patterns=list(_HF_MODEL_FILE_PATTERNS)))


def read_hf_pixel_stats(local_dir: str | Path) -> PixelStats:
    """Reads image normalization mean and standard deviation from preprocessor_config.json.

    Args:
        local_dir: Directory containing `preprocessor_config.json`.

    Returns:
        `PixelStats` named tuple containing channel means and standard deviations.

    Raises:
        FileNotFoundError: If `preprocessor_config.json` is missing.
        KeyError: If `image_mean` or `image_std` entries are missing.
    """
    path = Path(local_dir) / "preprocessor_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, so the normalization this checkpoint expects is unknown."
            f" Re-download it with `download_hf_model` (which fetches the preprocessor"
            f" config), or pass the statistics explicitly.")
    config = json.loads(path.read_text())
    try:
        return PixelStats(mean=tuple(config["image_mean"]), std=tuple(config["image_std"]))
    except KeyError as e:
        raise KeyError(f"{path} has no {e} entry, so it does not describe a normalization.") from e


def _read_scalar_config_value(config, name: str):
    value = getattr(config, name, None)
    return value[0] if isinstance(value, (tuple, list)) else value


class ViTEncoder(FrameEncoder):
    """Vision Transformer encoder backed by Hugging Face DiNOv2 checkpoints."""

    def __init__(self, *, local_dir: str):
        """Args:
        local_dir: Path to a directory containing config.json, preprocessor_config.json
            and the model weights.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(local_dir, local_files_only=True)
        self.patch_size = _read_scalar_config_value(self.config, "patch_size")
        self.image_size = _read_scalar_config_value(self.config, "image_size")
        self._set_pixel_stats(read_hf_pixel_stats(local_dir))

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=frames, output_hidden_states=False)
        tokens = outputs.last_hidden_state  # (B, 1 + N, D)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        feature_map = patch_tokens_to_feature_map(patch_tokens, frames, self.patch_size)
        return feature_map, cls_token


def dinov2_vit_encoder(
    variant: str = "dinov2_vits14",
    *,
    params_dir: str | None = None,
    **encoder_kwargs: Any,
) -> ViTEncoder:
    """Creates a DINOv2 vision transformer encoder.

    Args:
        variant: Model variant name (e.g. 'dinov2_vits14', 'dinov2_vitb14').
        params_dir: Local cache directory for downloaded model weights.
        **encoder_kwargs: Additional arguments forwarded to `ViTEncoder`.

    Returns:
        Configured `ViTEncoder` instance.
    """
    repo_id = DINOV2_VARIANT_TO_HF_REPO.get(variant)
    if repo_id is None:
        raise ValueError(f"Unsupported DiNOv2 variant '{variant}'.")
    if params_dir is None:
        raise ValueError("params_dir is required to cache DiNOv2 model files.")

    local_dir = download_hf_model(repo_id, params_dir)

    return ViTEncoder(local_dir=str(local_dir), **encoder_kwargs)
