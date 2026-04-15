"""
Gemma 4 predictor using vLLM for zero-shot road attribute classification.
"""

from pathlib import Path

from PIL import Image

from .base_vllm import BaseVLLMPredictor
from .gemma_utils import build_gemma_chat_messages


class Gemma4VLvLLMPredictor(BaseVLLMPredictor):
    """Zero-shot road attribute classifier using Gemma 4 with vLLM backend.

    The model is loaded lazily on first prediction.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-27b-it",
        gpu_memory_utilization: float = 0.80,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 8192,
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
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
            trust_remote_code=False,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )

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
