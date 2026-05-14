"""
Base class for HuggingFace transformers-backed VLM predictors.

Extracts shared model loading (env vars, flash-attention fallback, lazy init)
and generation logic (gen_kwargs, token slicing, thinking stripping) so that
model-specific subclasses only need to implement ``_load_hf_model()`` and
``_prepare_hf_inputs()``.
"""

import os
from abc import abstractmethod
from pathlib import Path

import torch
from PIL import Image

from .base import BaseVLMPredictor
from .thinking import strip_thinking


class BaseHFPredictor(BaseVLMPredictor):
    """Abstract HF-transformers predictor with shared loading and generation.

    Subclasses must implement:
    - ``_load_hf_model()`` to load the model and processor into
      ``self._model`` and ``self._processor``.
    - ``_prepare_hf_inputs()`` to tokenize a prompt+image pair into
      a dict-like object suitable for ``model.generate(**inputs)``.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        max_response_tokens: int = 512,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 15,
        min_new_tokens: int = 0,
        debug: bool = False,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_id=model_id,
            max_response_tokens=max_response_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.use_flash_attention = use_flash_attention

        # Lazy-loaded components
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Load the model and processor (called on first prediction)."""
        if self._model is not None and self._processor is not None:
            return

        # Defensive environment settings
        if "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        if "TRANSFORMERS_NO_TF" not in os.environ:
            os.environ["TRANSFORMERS_NO_TF"] = "1"

        class_name = type(self).__name__
        print(f"[{class_name}] Loading model: {self.model_id}")

        try:
            attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
            device_map = {"": self.device} if isinstance(self.device, (str, torch.device)) else "auto"
            self._load_hf_model(attn_impl, device_map)
        except Exception as e:
            self._model = None
            self._processor = None
            if self.use_flash_attention:
                raise RuntimeError(
                    "Model load failed with flash_attention_2 enabled. "
                    "Retry with use_flash_attention=False (CLI: --no-flash-attention). "
                    f"Original error: {e}"
                ) from e
            raise e

        print(f"[{class_name}] Model loaded successfully")

    @abstractmethod
    def _load_hf_model(self, attn_impl: str, device_map: dict | str) -> None:
        """Load the HF model and processor into self._model and self._processor.

        Called inside ``_load_model()`` after env-var setup. The caller handles
        the flash-attention error fallback, so this method should just attempt
        the load and let exceptions propagate.

        Args:
            attn_impl: Attention implementation string (e.g. "flash_attention_2").
            device_map: Device map for ``from_pretrained``.
        """

    @abstractmethod
    def _prepare_hf_inputs(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare tokenized inputs for the specific model family.

        Returns:
            A dict-like object (dict or BatchEncoding) that supports
            ``.to(device)``, has an ``input_ids`` key, and can be unpacked
            into ``model.generate(**inputs)``.
        """

    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> tuple[str, str | None]:
        """Generate response for a single prompt+image."""
        inputs = self._prepare_hf_inputs(pil_image, prompt)
        inputs = inputs.to(self._model.device)

        if self.debug:
            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}, Prompt chars: {len(prompt)}")

        with torch.no_grad():
            gen_kwargs = dict(max_response_tokens=self.max_response_tokens)
            if self.min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = self.min_new_tokens
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Get only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_response = self._processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        thinking_text = None
        if self.enable_thinking:
            raw_response, thinking_text = strip_thinking(raw_response)
            if self.debug and thinking_text:
                print(f"[DEBUG] Thinking ({len(thinking_text)} chars): {thinking_text[:300]}...")

        if self.debug:
            print(f"[DEBUG] Output tokens: {output_ids.shape[1]}, Response: {raw_response[:200]}...")

        return raw_response, thinking_text
