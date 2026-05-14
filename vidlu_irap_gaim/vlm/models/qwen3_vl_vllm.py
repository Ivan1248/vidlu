"""
Qwen3-VL predictor using vLLM for zero-shot road attribute classification.
"""

from pathlib import Path

from PIL import Image

from .base_vllm import BaseVLLMPredictor
from .qwen_utils import build_qwen_chat_messages


class Qwen3VLvLLMPredictor(BaseVLLMPredictor):
    """Zero-shot road attribute classifier using Qwen3-VL with vLLM backend.

    The model is loaded lazily on first prediction.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        gpu_memory_utilization: float = 0.80,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 8192,
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
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_response_tokens=max_response_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
            trust_remote_code=True,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )

    def _prepare_vllm_input(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare input in vLLM format using Qwen-VL utilities."""
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = build_qwen_chat_messages(pil_image, prompt)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
