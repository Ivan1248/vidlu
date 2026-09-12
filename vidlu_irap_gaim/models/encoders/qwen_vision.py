"""Qwen3-VL vision tower adapter for ImageSequenceClassifier.

Wraps the vision transformer tower of a Qwen3-VL model as a `FrameEncoder` with
PEFT LoRA fine-tuning and optional 4-bit quantization. Checkpoints store only LoRA
adapter parameters, restoring frozen base tower weights on load.
"""

import torch
from PIL import Image
from torch import nn

from vidlu_irap_gaim.peft_utils import (check_lora_match, make_nf4_quantization_config,
                                        normalize_lora_target_modules)

from .base import FrameEncoder, TrainabilityMode

_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# List → PEFT suffix match. Covers both Qwen2-VL vision naming ("qkv"/"proj") and
# Qwen3-VL-style separate projections, scoped to the extracted visual submodule so
# it never touches the language model.
_DEFAULT_LORA_TARGET_MODULES = ("qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj")

# Candidate attributes under which HF stores the vision tower, most specific first.
_VISION_TOWER_ATTR_PATHS = ("model.visual", "visual", "model.vision_model", "vision_model")


def _resolve_vision_tower(model: nn.Module) -> tuple[nn.Module, str]:
    for path in _VISION_TOWER_ATTR_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.Module):
            return obj, path
    available = [name for name, _ in model.named_children()]
    raise RuntimeError(
        f"Could not locate the Qwen vision tower on {type(model).__name__}. "
        f"Tried {_VISION_TOWER_ATTR_PATHS}; top-level children are {available}. "
        f"Update _VISION_TOWER_ATTR_PATHS."
    )


def _to_pil_uint8(frame: torch.Tensor) -> Image.Image:
    """Converts a single ``(3, H, W)`` tensor in ``[0, 1]`` to a uint8 RGB PIL image."""
    arr = (frame.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    return Image.fromarray(arr, mode="RGB")


class Qwen3VLVisionEncoder(FrameEncoder):
    """Qwen3-VL vision tower with LoRA, adapted to the encoder contract."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] | str = _DEFAULT_LORA_TARGET_MODULES,
        load_in_4bit: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_id = model_id
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.load_in_4bit = load_in_4bit
        self._device = device
        self._dtype = dtype
        self._build()
        # The image processor normalizes with Qwen's own statistics as part of
        # patchification, so frames are passed through unchanged.
        self._set_pixel_stats(None)

    def _build(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")

        full_model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=(make_nf4_quantization_config(self._dtype) if self.load_in_4bit
                                 else None),
            dtype=self._dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

        visual, resolved_path = _resolve_vision_tower(full_model)
        print(f"[Qwen3VLVisionEncoder] Using vision tower at '{resolved_path}'.")

        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=normalize_lora_target_modules(self.lora_target_modules),
            bias="none",
            task_type=None,  # feature extraction, not causal LM
        )
        self.visual = get_peft_model(visual, peft_config)
        check_lora_match(self.visual, self.lora_target_modules, type(self).__name__)
        trainable = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)

        # State-dict names of the tower entries that are checkpointed (see `state_dict`).
        # Captured here, where `requires_grad` still marks exactly the LoRA adapter, so
        # that later freezing/unfreezing cannot change what a checkpoint contains.
        self._adapter_state_keys = frozenset(
            f"visual.{name}" for name, p in self.visual.named_parameters() if p.requires_grad
        )

        # AutoProcessor is the single source of truth for patchification + Qwen's
        # image normalization, so only its image_processor is used here.
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._image_processor = getattr(processor, "image_processor", processor)
        # Fast (torchvision-based) processors accept torch tensors directly and can
        # run on GPU, avoiding the tensor->PIL->uint8 round-trip per batch.
        self._processor_is_fast = bool(getattr(self._image_processor, "is_fast", False))
        if not self._processor_is_fast:
            print(
                "[Qwen3VLVisionEncoder] Slow (PIL-based) image processor loaded; "
                "falling back to the per-frame PIL path (slower)."
            )
        self._merge_size = getattr(self.visual.config, "spatial_merge_size", None) \
            if hasattr(self.visual, "config") else None

        total = sum(p.numel() for p in self.visual.parameters())
        print(
            f"[Qwen3VLVisionEncoder] Loaded. Trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)."
        )

    def _tower_device(self) -> torch.device:
        return next(self.visual.parameters()).device

    def encode(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = frame.shape[0]
        if self._processor_is_fast:
            # Tensor fast path: feed the [0,1] float batch directly. `do_rescale=False`
            # tells the processor the values are already in [0,1] (it otherwise assumes
            # [0,255]). Resize/normalize/patchify run as torchvision tensor ops on the
            # frame's device – no PIL, no uint8 quantization.
            proc = self._image_processor(
                images=frame.clamp(0, 1), do_rescale=False, return_tensors="pt"
            )
        else:
            images = [_to_pil_uint8(frame[i]) for i in range(B)]
            proc = self._image_processor(images=images, return_tensors="pt")

        # Read while still on the CPU, so that the grid costs no device-to-host sync. The
        # spatial feature map needs a uniform grid across the batch, which holds for the
        # fixed-size frames this pipeline feeds. Fail loudly otherwise rather than fabricate
        # a map.
        grids = proc["image_grid_thw"].tolist()
        if any(g != grids[0] for g in grids):
            raise NotImplementedError(
                "Non-uniform vision grids across the batch are unsupported for the "
                f"spatial feature map (grids={grids}). Feed fixed-size frames.")
        merge = self._merge_size or 1
        t, h, w = grids[0]
        gh, gw = h // merge, w // merge
        num_tokens_per_image = t * gh * gw  # merged tokens: prod(grid_thw) / merge^2

        device = self._tower_device()
        pixel_values = proc["pixel_values"].to(device=device, dtype=self._dtype)
        grid_thw = proc["image_grid_thw"].to(device)
        out = self.visual(pixel_values, grid_thw=grid_thw)
        if isinstance(out, torch.Tensor):
            tokens = out  # older convention: forward returns the merged tokens directly
        else:
            # Current transformers returns a dataclass where `last_hidden_state` holds
            # the RAW pre-merge patch tokens and the spatially merged tokens (what the
            # LM consumes) are in `pooler_output`.
            tokens = getattr(out, "pooler_output", None)
            if tokens is None:
                tokens = getattr(out, "last_hidden_state", None)
            if tokens is None:
                raise RuntimeError(
                    f"Qwen vision tower returned {type(out).__name__} with neither "
                    f"pooler_output nor last_hidden_state. Inspect its output type."
                )
        # The tower computes in bf16. Features are returned in the caller's dtype so that
        # the fp32 heads (and non-AMP eval) compose without a dtype mismatch.
        tokens = tokens.to(frame.dtype)
        # tokens: (B * num_tokens_per_image, D)
        if tokens.shape[0] != B * num_tokens_per_image:
            raise RuntimeError(
                f"Merged-token count mismatch: tower returned {tokens.shape[0]} tokens, "
                f"expected {B * num_tokens_per_image} from grid_thw={grids} (merge={merge}). "
                f"Verify the grid_thw/merge convention for this transformers version."
            )

        # Views, not copies: the grid is uniform, so the rows split evenly between images.
        per_image_tokens = tokens.reshape(B, num_tokens_per_image, -1)
        pooled = per_image_tokens.mean(dim=1)  # (B, D)
        # Spatial feature map (only consumed when attention=True), temporal axis collapsed.
        feature_map = (per_image_tokens.reshape(B, t, gh, gw, -1)
                       .mean(dim=1)
                       .permute(0, 3, 1, 2))  # (B, D, gh, gw)
        return feature_map, pooled

    def lora_parameters(self) -> list[nn.Parameter]:
        # `_adapter_state_keys` was captured while `requires_grad` still marked exactly
        # the adapters, so it identifies them independently of the current trainability.
        return [p for name, p in self.named_parameters() if name in self._adapter_state_keys]

    def set_trainable(self, mode: TrainabilityMode) -> None:
        if mode == "all":
            raise ValueError(
                "Qwen3VLVisionEncoder cannot train its base tower: it is loaded frozen (and"
                " optionally 4-bit quantized), and only the LoRA adapters are checkpointed."
                " Use mode='lora'.")
        super().set_trainable(mode)

    # --- adapter-only checkpointing ---------------------------------------------------

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Extracts a state dictionary holding the LoRA adapter parameters, without the
        base tower weights."""
        if args:  # the deprecated positional form would bypass the filtering below
            raise TypeError(f"{type(self).__name__}.state_dict accepts only keyword arguments.")
        destination = super().state_dict(destination=destination, prefix=prefix,
                                         keep_vars=keep_vars)
        for key in [k for k in destination if k.startswith(f"{prefix}visual.")]:
            if key[len(prefix):] not in self._adapter_state_keys:
                del destination[key]
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        """Populates missing base tower parameters from the initialized model when an
        adapter-only checkpoint is loaded."""
        loadable = {name for name, _ in self.named_parameters()}
        loadable |= {name for name, _ in self.named_buffers()}
        for name, value in super().state_dict().items():
            if name in self._adapter_state_keys or not name.startswith("visual."):
                continue
            if name in loadable:
                state_dict[prefix + name] = value
            else:
                state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
