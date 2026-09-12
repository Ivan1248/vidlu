"""Shared pieces of LoRA / QLoRA wiring.

The VLM classifiers (`vlm.finetuning.model`) and the vision-tower encoders
(`models.encoders`) all attach PEFT adapters to a frozen, optionally 4-bit
quantized, Hugging Face model. The quantization recipe and the checks around
the adapter live here so that a change to one of them reaches every model.
"""

import torch
from torch import nn


def make_nf4_quantization_config(compute_dtype: torch.dtype = torch.bfloat16):
    """Builds the 4-bit NF4 `BitsAndBytesConfig` the base weights are loaded with (QLoRA)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )


def normalize_lora_target_modules(targets: str | tuple[str, ...] | list[str]) -> str | list[str]:
    """PEFT's two target conventions: a `str` is a regex (full-path match), a sequence is a
    list of suffixes to match module names against."""
    return targets if isinstance(targets, str) else list(targets)


def num_lora_parameters(module: nn.Module) -> int:
    """The number of adapter parameters PEFT injected into `module`."""
    return sum(p.numel() for name, p in module.named_parameters() if "lora_" in name)


def check_lora_match(module: nn.Module, targets, owner_name: str) -> None:
    """Raises when the LoRA target names matched no sublayer of `module`.

    PEFT then silently produces an adapter with zero parameters, so the run would train
    nothing. Important for new model families, whose sublayer names may differ from the
    default targets.
    """
    if num_lora_parameters(module) == 0:
        raise RuntimeError(
            f"[{owner_name}] LoRA wrapping produced 0 trainable parameters:"
            f" lora_target_modules={targets!r} matched nothing. Inspect the module names via"
            f" `{{name for name, _ in module.named_modules()}}` (or"
            f" `vidlu_irap_gaim.tools.dump_module_names` for a VLM) and update the targets.")
