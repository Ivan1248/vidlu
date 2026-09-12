"""
Qwen3-VL predictor using vLLM for zero-shot road attribute classification.
"""

from functools import partialmethod

from PIL import Image

from .base_vllm import BaseVLLMPredictor
from .qwen_utils import build_qwen_chat_messages


class Qwen3VLvLLMPredictor(BaseVLLMPredictor):
    """Zero-shot road attribute classifier using Qwen3-VL with vLLM backend.

    The model is loaded lazily on first prediction.
    """

    __init__ = partialmethod(BaseVLLMPredictor.__init__,
                             model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
                             trust_remote_code=True)

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
