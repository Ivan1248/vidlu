from typing import Any

import torch
from transformers import AutoConfig, AutoModel

from .base import FrameEncoder
from .vit import (
    _read_scalar_config_value,
    download_hf_model,
    patch_tokens_to_feature_map,
    read_hf_pixel_stats,
)

# Known Facebook MAE checkpoints (ImageNet-pretrained ViT encoders). Any HF repo id
# exposing a ViTMAEModel works; these are convenient aliases for `variant`.
MAE_VARIANT_TO_HF_REPO = {
    "vit-mae-base": "facebook/vit-mae-base",
    "vit-mae-large": "facebook/vit-mae-large",
    "vit-mae-huge": "facebook/vit-mae-huge",
}


class MAEEncoder(FrameEncoder):
    """Masked-Autoencoder ViT encoder backed by a Hugging Face ``ViTMAEModel``.

    Behaves as a plain ViT feature extractor: masking is disabled
    (``config.mask_ratio = 0``) so all patch tokens pass through in order.
    """

    def __init__(self, *, local_dir: str):
        """Args:
        local_dir: Path to a directory containing config.json, preprocessor_config.json
            and the weights of a ``ViTMAEModel`` checkpoint.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(local_dir, local_files_only=True)
        # Disable MAE random masking so the encoder sees (and returns) every patch
        # token, unshuffled — i.e. acts as a standard ViT feature extractor rather
        # than the pre-training masked encoder (default mask_ratio=0.75).
        self.config.mask_ratio = 0.0
        self.model = AutoModel.from_pretrained(
            local_dir, config=self.config, local_files_only=True
        )
        self.patch_size = _read_scalar_config_value(self.config, "patch_size")
        self.image_size = _read_scalar_config_value(self.config, "image_size")
        self._set_pixel_stats(read_hf_pixel_stats(local_dir))

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ViT-MAE has fixed-size position embeddings (unlike DINOv2 it does not
        # interpolate them by default), so non-224 inputs otherwise fail a strict
        # size check. interpolate_pos_encoding=True resizes the position embeddings
        # to the actual patch grid, keeping native resolution / aspect ratio.
        outputs = self.model(pixel_values=frames, interpolate_pos_encoding=True)
        tokens = outputs.last_hidden_state  # (B, 1 + N, D)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        feature_map = patch_tokens_to_feature_map(patch_tokens, frames, self.patch_size)
        return feature_map, cls_token


def mae_vit_encoder(
    variant: str = "facebook/vit-mae-base",
    *,
    params_dir: str | None = None,
    **encoder_kwargs: Any,
) -> MAEEncoder:
    """Factory for :class:`MAEEncoder`.

    Args:
        variant: Either a short alias (``"vit-mae-base"``/``-large``/``-huge``) or a
            full Hugging Face repo id (e.g. ``"facebook/vit-mae-base"``).
        params_dir: Directory to cache the downloaded config/weights in.
    """
    repo_id = MAE_VARIANT_TO_HF_REPO.get(variant, variant)
    if params_dir is None:
        raise ValueError("params_dir is required to cache MAE model files.")

    local_dir = download_hf_model(repo_id, params_dir)

    return MAEEncoder(local_dir=str(local_dir), **encoder_kwargs)
