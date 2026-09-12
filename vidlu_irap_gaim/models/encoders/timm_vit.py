"""timm-backed ViT backbones: SigLIP 2, SPAR, and DINOv3.

All three are plain timm vision transformers, so they share one encoder class and
differ only in the model name and (for SPAR) the source of the weights:

=========================  =========================================  ====  ====================
Backbone                   timm model                                 dim   Normalization
=========================  =========================================  ====  ====================
SigLIP 2 ViT-B/16 @512     ``vit_base_patch16_siglip_512.v2_webli``   768   mean = std = 0.5
SPAR ALL SigLIP2 ViT-B/16  same module, weights from a SPAR file      768   mean = std = 0.5
DINOv3 ViT-L/16            ``vit_large_patch16_dinov3.lvd1689m``      1024  ImageNet
=========================  =========================================  ====  ====================

Normalization is read from the checkpoint's own recorded preprocessing via
``timm.data.resolve_model_data_config`` rather than being chosen by the caller, and is
applied inside the encoder (see :mod:`.base`). The values above are what that call
returns; they agree with the upstream configs:

- SigLIP 2: ``mean = std = [0.5, 0.5, 0.5]``, bicubic, patch 16, width 768 --
  https://huggingface.co/timm/ViT-B-16-SigLIP2-512/raw/main/open_clip_config.json
- DINOv3 (LVD-1689M): ImageNet statistics ``mean = (0.485, 0.456, 0.406)``,
  ``std = (0.229, 0.224, 0.225)`` -- https://github.com/facebookresearch/dinov3

Frames are fed at their native size (iRAP: 288x384, i.e. an 18x24 patch grid) rather
than resized to the pretraining resolution. ``dynamic_img_size=True`` makes timm
resample the position embeddings to the actual grid; DINOv3 uses RoPE and has no
position-embedding table at all, so it is resolution-agnostic by construction. Pass
``input_size`` to resize instead -- see :class:`TimmViTEncoder`.

SPAR
----
SPAR ("Single-Pass Any-Resolution ViT for Open-vocabulary Segmentation", Kombol,
Martinovic, Segvic, Tolias, CVPR 2026; https://github.com/naomikombol/SPAR,
https://arxiv.org/abs/2604.02252) distills a finely-strided sliding-window teacher into
a single-pass student, which makes the backbone markedly better behaved away from its
native resolution -- the regime this pipeline runs in. The "ALL" variant trains every
parameter (SPAR's default unfreezes only the last two blocks), and is distilled on 25k
SA-1B images from a stride-24 SigLIP2 ViT-B/16 @512 teacher.

Its published weights are Google Drive files, not a hub repo, so the checkpoint path is
passed explicitly. They are open_clip *wrapper* state dicts, of which only the ViT trunk
is relevant here; :func:`load_spar_trunk_state_dict` extracts it.
"""

import re
import typing as T
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vidlu_irap_gaim.peft_utils import check_lora_match

from .base import FrameEncoder, PixelStats
from .vit import patch_tokens_to_feature_map

#: Convenient aliases for `model_name`. Any timm ViT name also works.
SIGLIP2_VARIANT_TO_TIMM_MODEL = {
    "base-224": "vit_base_patch16_siglip_224.v2_webli",
    "base-256": "vit_base_patch16_siglip_256.v2_webli",
    "base-384": "vit_base_patch16_siglip_384.v2_webli",
    "base-512": "vit_base_patch16_siglip_512.v2_webli",
    "large-256": "vit_large_patch16_siglip_256.v2_webli",
    "large-384": "vit_large_patch16_siglip_384.v2_webli",
    "large-512": "vit_large_patch16_siglip_512.v2_webli",
}

DINOV3_VARIANT_TO_TIMM_MODEL = {
    "small": "vit_small_patch16_dinov3.lvd1689m",
    "base": "vit_base_patch16_dinov3.lvd1689m",
    "large": "vit_large_patch16_dinov3.lvd1689m",
    # Satellite-pretrained. These use different pixel statistics, which
    # `resolve_model_data_config` picks up automatically.
    "large-sat": "vit_large_patch16_dinov3.sat493m",
}

# LoRA is injected into the transformer blocks only, leaving the patch embedding and the
# pooling head alone. `attn.proj` and `attn_pool.proj` would both match a bare "proj"
# suffix, so the targets are given as a full-match regex over module names.
_LORA_TARGET_MODULES_RE = r"blocks\.\d+\.(attn\.(qkv|q_proj|k_proj|v_proj|proj)|mlp\.(fc1|fc2))"


class TimmViTEncoder(FrameEncoder):
    """A timm vision transformer adapted to the :class:`FrameEncoder` contract.

    Args:
        model_name: A timm model name, normally including a pretrained tag
            (e.g. ``'vit_base_patch16_siglip_512.v2_webli'``).
        pretrained: Whether to download and load timm's pretrained weights. Set to
            ``False`` when `params_path` supplies the weights instead.
        params_path: Optional path to a SPAR checkpoint whose ViT trunk replaces the
            backbone weights. See :func:`load_spar_trunk_state_dict`.
        input_size: ``(height, width)`` (or a single int) to resize frames to before
            encoding, or ``None`` (default) to encode at the frames' native size.
            Resizing does not preserve the aspect ratio, matching SigLIP's own
            ``resize_mode: squash`` convention.
        pooling: How the per-frame vector is formed. ``'default'`` uses the model's own
            head (SigLIP's attention pooling, DINOv3's patch-token average).
            ``'cls+avg'`` concatenates the class token with the average of the patch
            tokens, doubling the pooled width; this is the feature DINOv3's own linear
            evaluation uses (`dinov3/eval/linear.py`), and it needs a class token.
        grad_checkpointing: Trade compute for activation memory. Needed to fine-tune
            ViT-L on sequences of frames at a useful batch size.
        num_trainable_blocks: How many trailing transformer blocks
            ``set_trainable('last_blocks')`` unfreezes.
        lora_r: If given, inject LoRA adapters of this rank into the transformer blocks,
            enabling ``set_trainable('lora')``. Requires ``peft``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool = True,
        params_path: str | Path | None = None,
        input_size: int | tuple[int, int] | None = None,
        pooling: str = "default",
        grad_checkpointing: bool = False,
        num_trainable_blocks: int = 2,
        lora_r: int | None = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **timm_kwargs: T.Any,
    ):
        super().__init__()
        import timm

        self.model_name = model_name
        self.input_size = ((input_size, input_size) if isinstance(input_size, int)
                           else None if input_size is None else tuple(input_size))
        if pooling not in ("default", "cls+avg"):
            raise ValueError(f"{pooling=} is not one of ('default', 'cls+avg').")
        self.pooling = pooling
        self.num_trainable_blocks = num_trainable_blocks

        # `num_classes=0` turns the classifier into an identity, so `forward_head`
        # returns the pooled pre-logits feature. `dynamic_img_size=True` resamples the
        # position embeddings to whatever patch grid the input produces.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                       dynamic_img_size=True, **timm_kwargs)
        if pooling == "cls+avg" and self.model.num_prefix_tokens == 0:
            raise ValueError(
                f"pooling='cls+avg' needs a class token, but {model_name} has no prefix"
                f" tokens (it pools with an attention head). Use pooling='default'.")
        if params_path is not None:
            self._load_spar_params(params_path)
        if grad_checkpointing:
            self.model.set_grad_checkpointing()

        patch_size = self.model.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size

        data_config = timm.data.resolve_model_data_config(self.model)
        self._set_pixel_stats(PixelStats(mean=tuple(data_config["mean"]),
                                         std=tuple(data_config["std"])))
        self.pretraining_input_size = tuple(data_config["input_size"][1:])

        self._lora_r = lora_r
        if lora_r is not None:
            self._inject_lora(lora_r, lora_alpha, lora_dropout)

    # Contract #####################################################################

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_size is not None and frames.shape[-2:] != self.input_size:
            # `antialias=True` matters when downsampling, and bicubic matches the
            # interpolation recorded in the checkpoints' data configs.
            frames = F.interpolate(frames, size=self.input_size, mode="bicubic",
                                   align_corners=False, antialias=True)
        tokens = self.model.forward_features(frames)
        # SigLIP pools with an attention head over all tokens and has no prefix tokens.
        # DINOv3 prepends a CLS token and 4 register tokens. Neither belongs in the map.
        patch_tokens = tokens[:, self.model.num_prefix_tokens:]
        if self.pooling == "cls+avg":
            # `forward_features` has already applied the final norm, so these are the
            # same tokens DINOv3's linear evaluation concatenates.
            pooled = torch.cat([tokens[:, 0], patch_tokens.mean(dim=1)], dim=-1)
        else:
            pooled = self.model.forward_head(tokens)
        feature_map = patch_tokens_to_feature_map(patch_tokens, frames, self.patch_size)
        return feature_map, pooled

    def pool_parameters(self) -> list[nn.Parameter]:
        # `forward_head` applies `attn_pool` (only for attention pooling, e.g. SigLIP's MAP
        # head) and then `fc_norm`. A backbone pooling with a CLS token or an average has no
        # parameters here, which makes 'pool' equivalent to 'none' for it. timm sets
        # `attn_pool` to None rather than omitting it, so `getattr`'s default is not enough.
        modules = (getattr(self.model, name, None) for name in ("attn_pool", "fc_norm"))
        return [p for m in modules if m is not None for p in m.parameters()]

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.model.named_parameters() if "lora_" in name]

    def last_block_parameters(self) -> list[nn.Parameter]:
        """The last `num_trainable_blocks` transformer blocks, the output norm, and the
        pooling head – everything downstream of the frozen part."""
        if self.num_trainable_blocks > len(self.model.blocks):
            raise ValueError(
                f"num_trainable_blocks={self.num_trainable_blocks} exceeds the"
                f" {len(self.model.blocks)} blocks of {self.model_name}.")
        modules = list(self.model.blocks[len(self.model.blocks) - self.num_trainable_blocks:])
        if self.model.norm is not None:
            modules.append(self.model.norm)
        return [p for m in modules for p in m.parameters()] + self.pool_parameters()

    def param_groups(self, *, lr: float, weight_decay: float,
                     layer_decay: float | None = None) -> list[dict]:
        """Constructs parameter groups with optional layer-wise learning rate decay.

        Args:
            lr: Base learning rate.
            weight_decay: Weight decay value.
            layer_decay: Optional per-block learning rate multiplicative decay factor.

        Returns:
            List of parameter group dictionaries compatible with PyTorch optimizers.
        """
        if layer_decay is None:
            return super().param_groups(lr=lr, weight_decay=weight_decay)

        from timm.optim import param_groups_layer_decay

        groups = param_groups_layer_decay(
            self.model, weight_decay=weight_decay, layer_decay=layer_decay,
            no_weight_decay_list=self.model.no_weight_decay())
        # timm reports the depth scaling as `lr_scale` and leaves applying it to the
        # caller, so resolve it into an explicit per-group `lr` here. Groups holding no
        # trainable parameter are dropped so that e.g. LoRA mode yields adapter groups only.
        resolved = []
        for group in groups:
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            resolved.append(dict(params=params, lr=lr * group["lr_scale"],
                                 weight_decay=group["weight_decay"]))
        if not resolved:
            raise RuntimeError(
                f"{type(self).__name__}.param_groups produced no trainable parameters"
                f" (trainability={self.trainability!r}).")
        return resolved

    # Construction helpers #########################################################

    def _inject_lora(self, r: int, alpha: int, dropout: float) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                            target_modules=_LORA_TARGET_MODULES_RE, bias="none")
        # `inject_adapter_in_model` mutates the module tree in place, unlike
        # `get_peft_model`, which wraps it. That keeps `self.model` a timm
        # `VisionTransformer`, so `forward_features`, `num_prefix_tokens` and
        # `group_matcher` (needed for layer-wise LR decay) keep working.
        inject_adapter_in_model(config, self.model)
        check_lora_match(self.model, _LORA_TARGET_MODULES_RE, self.model_name)

    def _load_spar_params(self, params_path: str | Path) -> None:
        state_dict = load_spar_trunk_state_dict(params_path)
        # Strict: SPAR's trunk is exactly `vit_base_patch16_siglip_512`, so any missing or
        # unexpected key means the extraction was wrong and the run would silently train a
        # partly randomly-initialized backbone.
        self.model.load_state_dict(state_dict, strict=True)


def load_spar_trunk_state_dict(params_path: str | Path) -> dict[str, torch.Tensor]:
    """Extracts timm ViT trunk parameters from a SPAR checkpoint.

    Args:
        params_path: Path to the SPAR `.pt`, `.pth`, or `.ckpt` file.

    Returns:
        State dictionary matching the module hierarchy of `VisionTransformer`.

    Raises:
        RuntimeError: If no recognized trunk prefix is present in the checkpoint.
    """
    checkpoint = torch.load(Path(params_path).expanduser(), map_location="cpu",
                            weights_only=True)
    for key in ("state_dict", "model", "module"):
        if isinstance(checkpoint, dict) and key in checkpoint \
                and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    # Most specific first: strip everything up to and including the timm trunk.
    for prefix in ("segmentor.net.model.visual.trunk.", "model.visual.trunk.",
                   "net.model.visual.trunk.", "visual.trunk.", "trunk."):
        trunk = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        if trunk:
            return trunk
    if any(k.startswith("patch_embed.") for k in checkpoint):
        return dict(checkpoint)  # already a bare timm trunk

    observed = sorted({re.split(r"\.\d+\.|\.", k, maxsplit=1)[0] for k in checkpoint})
    raise RuntimeError(
        f"Could not locate a timm ViT trunk in the SPAR checkpoint '{params_path}'."
        f" Top-level key prefixes are {observed[:20]}. Example keys:"
        f" {list(checkpoint)[:5]}. Extend the prefix list in `load_spar_trunk_state_dict`.")


def siglip2_vit_encoder(variant: str = "base-512", *, spar_params_path: str | Path | None = None,
                        **encoder_kwargs: T.Any) -> TimmViTEncoder:
    """Creates a SigLIP 2 ViT encoder, optionally initialized with SPAR weights.

    Args:
        variant: SigLIP 2 model variant name or key in `SIGLIP2_VARIANT_TO_TIMM_MODEL`.
        spar_params_path: Optional path to a local SPAR checkpoint.
        **encoder_kwargs: Additional arguments forwarded to `TimmViTEncoder`.

    Returns:
        Configured `TimmViTEncoder` instance.
    """
    model_name = SIGLIP2_VARIANT_TO_TIMM_MODEL.get(variant, variant)
    if spar_params_path is not None:
        encoder_kwargs.setdefault("pretrained", False)
    return TimmViTEncoder(model_name, params_path=spar_params_path, **encoder_kwargs)


def dinov3_vit_encoder(variant: str = "large", *, pooling: str = "cls+avg",
                       **encoder_kwargs: T.Any) -> TimmViTEncoder:
    """Creates a DINOv3 vision transformer encoder.

    Args:
        variant: DINOv3 model variant name or key in `DINOV3_VARIANT_TO_TIMM_MODEL`.
        pooling: Feature pooling strategy ('cls+avg' or 'default'). See `TimmViTEncoder`.
        **encoder_kwargs: Additional arguments forwarded to `TimmViTEncoder`.

    Returns:
        Configured `TimmViTEncoder` instance.
    """
    return TimmViTEncoder(DINOV3_VARIANT_TO_TIMM_MODEL.get(variant, variant), pooling=pooling,
                          **encoder_kwargs)
