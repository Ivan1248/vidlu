"""
Gemma 4 predictor for zero-shot road attribute classification.
"""

from PIL import Image

from .base_hf import BaseHFPredictor
from .gemma_utils import build_gemma_chat_messages


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

    def _prepare_hf_inputs(self, pil_image: Image.Image, prompt: str) -> dict:
        # Image is embedded in messages by build_gemma_chat_messages;
        # do NOT pass images= (transformers >= 5.5 extracts from messages
        # and a second images kwarg raises "got multiple values").
        messages = build_gemma_chat_messages(pil_image, prompt)
        return self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
