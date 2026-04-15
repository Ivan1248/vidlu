"""
Gemma 4 predictor for zero-shot road attribute classification.
"""

import os
from pathlib import Path

import torch
from PIL import Image

from vidlu_irap_gaim.vlm.base import BaseVLMPredictor, strip_thinking
from vidlu_irap_gaim.vlm.gemma_utils import build_gemma_chat_messages


class Gemma4VLPredictor(BaseVLMPredictor):
    """Zero-shot road attribute classifier using Gemma 4.

    The model is loaded lazily on first prediction to avoid VRAM allocation
    during setup/import.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-27b-it",
        device: str | torch.device = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        max_new_tokens: int = 512,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 15,
        min_new_tokens: int = 0,
        debug: bool = False,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
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

        print(f"[Gemma4VLPredictor] Loading model: {self.model_id}")

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        try:
            attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

            device_map = {"": self.device} if isinstance(self.device, (str, torch.device)) else "auto"

            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)
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

        print("[Gemma4VLPredictor] Model loaded successfully")

    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> str:
        """Generate response for a single prompt+image."""
        messages = build_gemma_chat_messages(pil_image, prompt)

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            images=[pil_image],
            enable_thinking=self.enable_thinking,
        ).to(self._model.device)

        if self.debug:
            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}, Prompt chars: {len(prompt)}")

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
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

        if self.enable_thinking:
            raw_response, thinking_text = strip_thinking(raw_response)
            if self.debug and thinking_text:
                print(f"[DEBUG] Thinking ({len(thinking_text)} chars): {thinking_text[:300]}...")

        if self.debug:
            print(f"[DEBUG] Output tokens: {output_ids.shape[1]}, Response: {raw_response[:200]}...")

        return raw_response
