"""
Qwen3-VL predictor for zero-shot road attribute classification.
"""

import os
from pathlib import Path

import torch
from PIL import Image

from vidlu_irap_gaim.vlm.base import BaseVLMPredictor
from vidlu_irap_gaim.vlm.qwen_utils import build_qwen_chat_messages


class Qwen3VLPredictor(BaseVLMPredictor):
    """Zero-shot road attribute classifier using Qwen3-VL.

    The model is loaded lazily on first prediction to avoid VRAM allocation
    during setup/import.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str | torch.device = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        max_new_tokens: int = 512,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 15,
        min_new_tokens: int = 0,
        debug: bool = False,
    ):
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
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

        print(f"[Qwen3VLPredictor] Loading model: {self.model_id}")

        from transformers import AutoModelForVision2Seq, AutoProcessor

        try:
            attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
            
            # Map target device
            device_map = {"": self.device} if isinstance(self.device, (str, torch.device)) else "auto"

            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
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

        print("[Qwen3VLPredictor] Model loaded successfully")

    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> str:
        """Generate response for a single prompt+image."""
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = build_qwen_chat_messages(pil_image, prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        if self.debug:
            print(f"[DEBUG] Input shape: {inputs.input_ids.shape}, Prompt chars: {len(prompt)}")

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
            if self.min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = self.min_new_tokens
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Get only the newly generated tokens
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_response = self._processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        if self.debug:
            print(f"[DEBUG] Output tokens: {output_ids.shape[1]}, Response: {raw_response[:200]}...")

        return raw_response
