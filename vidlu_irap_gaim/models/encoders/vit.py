from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from vidlu.utils.misc import download_if_not_downloaded

DINOV2_VARIANT_TO_HF_REPO = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}

# Files required to load a HF ViT-style model (config + weights) locally
_HF_MODEL_FILES = ["config.json", "model.safetensors"]


def patch_tokens_to_feature_map(
    patch_tokens: torch.Tensor,
    frame: torch.Tensor,
    patch_size: int | None,
) -> torch.Tensor:
    """Reshape a `(B, N, D)` sequence of patch tokens into a `(B, D, grid_h, grid_w)`
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


class ViTEncoder(nn.Module):
    """Vision Transformer encoder backed by Hugging Face DiNOv2 checkpoints."""

    def __init__(self, *, local_dir: str):
        """Args:
        local_dir: Path to directory containing config.json and model.safetensors
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(local_dir, local_files_only=True)
        patch_size = getattr(self.config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        image_size = getattr(self.config, "image_size", None)
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]
        self.image_size = image_size

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=frame, output_hidden_states=False)
        tokens = outputs.last_hidden_state  # (B, 1 + N, D)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        feature_map = patch_tokens_to_feature_map(patch_tokens, frame, self.patch_size)
        return feature_map, cls_token

    def pooling_parameters(self) -> list[nn.Parameter]:
        return []


def _download_hf_model_files(repo_id: str, params_dir: str | Path) -> Path:
    params_dir = Path(params_dir).expanduser().resolve()
    local_dir = params_dir / repo_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in _HF_MODEL_FILES:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        local_path = local_dir / filename
        download_if_not_downloaded(url, local_path)

    return local_dir


def dinov2_vit_encoder(
    variant: str = "dinov2_vits14",
    *,
    params_dir: str | None = None,
    **encoder_kwargs: Any,
) -> ViTEncoder:
    repo_id = DINOV2_VARIANT_TO_HF_REPO.get(variant)
    if repo_id is None:
        raise ValueError(f"Unsupported DiNOv2 variant '{variant}'.")
    if params_dir is None:
        raise ValueError("params_dir is required to cache DiNOv2 model files.")

    local_dir = _download_hf_model_files(repo_id, params_dir)

    return ViTEncoder(local_dir=str(local_dir), **encoder_kwargs)

