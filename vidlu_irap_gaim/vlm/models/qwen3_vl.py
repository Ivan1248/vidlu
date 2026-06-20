"""
Qwen3-VL predictor for zero-shot road attribute classification.
"""

from PIL import Image

from .base_hf import BaseHFPredictor
from .qwen_utils import build_qwen_chat_messages


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

    def _prepare_hf_inputs(self, pil_image: Image.Image, prompt: str) -> dict:
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = build_qwen_chat_messages(pil_image, prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        return self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
