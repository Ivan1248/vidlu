"""
Gemma 4 predictor for zero-shot road attribute classification.
"""

from typing import Sequence

from PIL import Image

from .base_hf import BaseHFPredictor
from .gemma_utils import build_gemma_generation_inputs


class Gemma4VLPredictor(BaseHFPredictor):
    """Zero-shot road attribute classifier using Gemma 4.

    The model is loaded lazily on first prediction to avoid VRAM allocation
    during setup/import.
    """

    def __init__(self, model_id: str = "google/gemma-4-27b-it", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def _load_hf_model(self, attn_impl: str, device_map: dict | str) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def _build_generation_inputs(self, pil_images: Sequence[Image.Image], prompt: str) -> dict:
        return build_gemma_generation_inputs(
            self._processor, pil_images, [prompt] * len(pil_images),
            template_kwargs={"enable_thinking": self.enable_thinking})
