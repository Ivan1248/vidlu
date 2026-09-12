"""
Qwen3-VL predictor for zero-shot road attribute classification.
"""

from typing import Sequence

from PIL import Image

from .base_hf import BaseHFPredictor
from .qwen_utils import build_qwen_generation_inputs


class Qwen3VLPredictor(BaseHFPredictor):
    """Zero-shot road attribute classifier using Qwen3-VL.

    The model is loaded lazily on first prediction to avoid VRAM allocation
    during setup/import.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def _load_hf_model(self, attn_impl: str, device_map: dict | str) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def _build_generation_inputs(self, pil_images: Sequence[Image.Image], prompt: str) -> dict:
        return build_qwen_generation_inputs(
            self._processor, pil_images, [prompt] * len(pil_images),
            template_kwargs={"enable_thinking": self.enable_thinking})
