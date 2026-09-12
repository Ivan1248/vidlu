"""
Gemma 4 predictor using vLLM for zero-shot road attribute classification.
"""

from functools import partialmethod

from PIL import Image

from .base_vllm import BaseVLLMPredictor
from .gemma_utils import build_gemma_chat_messages


class Gemma4VLvLLMPredictor(BaseVLLMPredictor):
    """Zero-shot road attribute classifier using Gemma 4 with vLLM backend.

    The model is loaded lazily on first prediction.
    """

    __init__ = partialmethod(BaseVLLMPredictor.__init__,
                             model_id="google/gemma-4-27b-it",
                             trust_remote_code=False)

    def _prepare_vllm_input(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare input in vLLM format for Gemma 4."""
        messages = build_gemma_chat_messages(pil_image, prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        return {
            "prompt": text,
            "multi_modal_data": {"image": [pil_image]},
        }
