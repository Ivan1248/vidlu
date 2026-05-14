"""
Qwen3-VL classifier wrapper for Vidlu training integration.

Key design decisions:
- Eager loading in initialize() so parameters exist before optimizer creation
- Adapter-only state_dict (saves ~100MB instead of ~16GB)
- Single-device loading (no device_map="auto" for DDP compatibility)
- input_adapter accepted but unused (VLM inputs are tokenized, not image-space)
"""

from typing import Any
import os

import torch
from torch import nn


class Qwen3VLClassifier(nn.Module):
    """Qwen3-VL wrapper with LoRA for Vidlu integration.

    The model is loaded eagerly in initialize(), which Vidlu calls during
    build_and_init_model() -- before the Trainer creates the optimizer.
    This ensures all LoRA parameters are discoverable by model.parameters().

    Args:
        model_id: HuggingFace model ID for Qwen3-VL.
        lora_r: LoRA rank (default 64).
        lora_alpha: LoRA alpha scaling (default 128).
        lora_dropout: LoRA dropout rate (default 0.05).
        lora_target_modules: Modules to apply LoRA to.
        load_in_4bit: Whether to use 4-bit quantization.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        input_adapter: Accepted for Vidlu factory compatibility (unused).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        load_in_4bit: bool = True,
        use_gradient_checkpointing: bool = True,
        input_adapter=None,
    ):
        super().__init__()
        self.model_id = model_id
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.load_in_4bit = load_in_4bit
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Accepted for Vidlu factory compatibility. VLM tokenized inputs
        # should not be transformed by image-space input adapters.
        self.input_adapter = input_adapter

        # Placeholder so Vidlu can detect device before model loading
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    def initialize(self, init_input):
        """Eagerly load the model during Vidlu setup.

        Called by Vidlu's build_and_init_model() after model.to(device).
        Loads the HuggingFace model, applies LoRA, and sets up the processor
        so that all trainable parameters are available when the Trainer
        creates the optimizer.
        """
        self._load()

    def _load(self):
        """Load model, processor, and apply LoRA."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        if self._loaded:
            return

        device = self._device or "cuda"

        if "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        print(f"[Qwen3VLClassifier] Loading {self.model_id}...")

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        # Load to specific device (not device_map="auto" for DDP compatibility).
        # attn_implementation: Flash Attention 2 reduces memory bandwidth; fallback to sdpa/eager if unavailable.
        load_kwargs = dict(
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                load_kwargs["attn_implementation"] = attn_impl
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )
                if attn_impl != "eager":
                    print(f"[Qwen3VLClassifier] Using attention: {attn_impl}")
                break
            except Exception:
                if attn_impl == "eager":
                    raise
                load_kwargs.pop("attn_implementation", None)

        # Apply LoRA
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, peft_config)

        if self.use_gradient_checkpointing:
            # use_cache must be False with gradient checkpointing; set explicitly to avoid the warning
            self._model.config.use_cache = False
            self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        self._loaded = True

        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"[Qwen3VLClassifier] Loaded. Trainable: {trainable_params:,} / "
            f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass -- returns loss if labels provided, else logits."""
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Ensure initialize() was called "
                "(this happens automatically in Vidlu's build_and_init_model)."
            )
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )

    def generate(self, **kwargs):
        """Generate responses using the underlying model."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return self._model.generate(**kwargs)

    def to(self, device, *args, **kwargs):
        """Track device for loading."""
        self._device = device
        return super().to(device, *args, **kwargs)

    # --- Adapter-only checkpointing ---

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Return only LoRA adapter weights + config metadata.

        This reduces checkpoint size from ~16GB to ~100MB.
        """
        if not self._loaded:
            return {"_config": self._get_config_dict()}

        # Get only adapter state dict from PEFT (parameters with requires_grad=True)
        adapter_state = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                adapter_state[name] = param.data.cpu()

        return {
            "_config": self._get_config_dict(),
            "_adapter_state": adapter_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Load adapter weights from checkpoint."""
        if "_config" not in state_dict:
            raise ValueError("Invalid checkpoint: missing _config")

        ckpt_model_id = state_dict["_config"]["model_id"]
        if ckpt_model_id != self.model_id:
            raise ValueError(
                f"Model ID mismatch: checkpoint has '{ckpt_model_id}', but model expects '{self.model_id}'"
            )

        if "_adapter_state" in state_dict:
            self._load()
            adapter_state = state_dict["_adapter_state"]
            model_state = self._model.state_dict()
            loaded_count = 0
            for name, param in adapter_state.items():
                if name in model_state:
                    model_state[name].copy_(param.to(model_state[name].device))
                    loaded_count += 1
                elif strict:
                    raise KeyError(f"Missing key in model state dict: {name}")
            print(f"[Qwen3VLClassifier] Loaded {loaded_count} adapter parameters from checkpoint")

    def _get_config_dict(self) -> dict[str, Any]:
        """Get configuration dictionary for checkpointing."""
        return {
            "model_id": self.model_id,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
        }

    @property
    def processor(self):
        """Get the HuggingFace processor (tokenizer + image processor)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return self._processor

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only LoRA parameters for optimizer."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return [p for p in self._model.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if self._model is not None:
            self._model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
