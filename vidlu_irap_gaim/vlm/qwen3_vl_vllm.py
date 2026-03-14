"""
Qwen3-VL predictor using vLLM for zero-shot road attribute classification.
"""

import os
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .base import (
    BaseVLMPredictor,
    VLMPredictionResult,
    _to_pil_image,
)
from .prompts import DEFAULT_DETAIL_LEVEL, DetailLevel
from .qwen_utils import build_qwen_chat_messages

class Qwen3VLvLLMPredictor(BaseVLMPredictor):
    """Zero-shot road attribute classifier using Qwen3-VL with vLLM backend.

    The model is loaded lazily on first prediction.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        gpu_memory_utilization: float = 0.80,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 8192,
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
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = int(max_model_len)

        # Lazy-loaded components
        self._llm = None
        self._processor = None
        self._sampling_params = None

    def _load_model(self) -> None:
        """Load the vLLM engine and processor (called on first prediction)."""
        if self._llm is not None and self._processor is not None:
            return

        # vLLM multiprocessing setup
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        print(f"[Qwen3VLvLLMPredictor] Loading model with vLLM: {self.model_id}")

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        # Determine tensor parallel size
        tp_size = self.tensor_parallel_size
        if tp_size is None:
            tp_size = torch.cuda.device_count()
            print(f"[Qwen3VLvLLMPredictor] Using {tp_size} GPU(s) for tensor parallelism")

        # Initialize vLLM engine
        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            max_model_len=self.max_model_len,
            seed=0,
            enable_prefix_caching=True,
        )

        # Load processor for tokenization
        self._processor = AutoProcessor.from_pretrained(self.model_id)

        # Set up sampling parameters
        self._sampling_params = SamplingParams(
            temperature=0,  # Greedy decoding for reproducibility
            max_tokens=self.max_new_tokens,
            # vLLM expects an int here; None crashes in some versions.
            min_tokens=self.min_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )

        print("[Qwen3VLvLLMPredictor] Model loaded successfully")

    def _prepare_vllm_input(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare input in vLLM format."""
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = build_qwen_chat_messages(pil_image, prompt)
        # Apply chat template
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process vision info (qwen_vl_utils 0.0.14+ required for Qwen3-VL)
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

    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> str:
        """Generate response for a single prompt+image."""
        vllm_input = self._prepare_vllm_input(pil_image, prompt)

        if self.debug:
            print(f"[DEBUG] Prompt chars: {len(prompt)}")

        outputs = self._llm.generate([vllm_input], self._sampling_params, use_tqdm=False)
        raw_response = outputs[0].outputs[0].text

        if self.debug:
            print(f"[DEBUG] Response chars: {len(raw_response)}")
            print(f"[DEBUG] Response preview: {raw_response[:200]}...")

        return raw_response

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attrs_to_include: Sequence[str] | None = None,
        *,
        detail_level: DetailLevel = DEFAULT_DETAIL_LEVEL,
    ) -> list[VLMPredictionResult]:
        """Batched prediction using single vLLM generate() call per attribute chunk.

        This override processes all images in parallel for each attribute chunk,
        significantly improving GPU utilization compared to sequential processing.
        The response format is determined by the ``response_scheme`` passed to
        the constructor (default: StandardResponseScheme).

        Args:
            images: Sequence of PIL images.
            attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
            attrs_to_include: Subset of attributes to classify.
            detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").

        Returns:
            List of VLMPredictionResult, one per image.
        """
        self._load_model()

        # Convert all images to PIL format
        pil_images = [_to_pil_image(img) for img in images]
        num_images = len(pil_images)

        if attrs_to_include is None:
            attrs_to_include = list(attr_to_value_to_class_idx.keys())

        # Split attributes into chunks to avoid context overflow
        attrs_list = list(attrs_to_include)
        chunks = [attrs_list[i : i + self.chunk_size] for i in range(0, len(attrs_list), self.chunk_size)]

        if self.debug:
            print(f"[DEBUG] Batched inference: {num_images} images, {len(chunks)} attribute chunks")

        response_scheme = self._get_response_scheme(attr_to_value_to_class_idx)

        # Initialize result accumulators for each image
        all_predictions_per_image: list[dict] = [{} for _ in range(num_images)]
        all_prompts_per_image: list[list[str]] = [[] for _ in range(num_images)]
        all_responses_per_image: list[list[str]] = [[] for _ in range(num_images)]

        # Process each attribute chunk with batched image inference
        for chunk_idx, chunk_attrs in enumerate(chunks):
            if self.debug:
                print(
                    f"[DEBUG] Processing chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_attrs)} attrs, {num_images} images"
                )

            # Build prompt for this chunk (same for all images)
            prompt = response_scheme.build_prompt(chunk_attrs, detail_level=detail_level)

            # Build vLLM inputs for all images with this prompt
            vllm_inputs = [self._prepare_vllm_input(img, prompt) for img in pil_images]

            # Single batched generate() call for all images
            outputs = self._llm.generate(vllm_inputs, self._sampling_params, use_tqdm=False)

            # Parse results for each image
            for img_idx, output in enumerate(outputs):
                raw_response = output.outputs[0].text
                all_responses_per_image[img_idx].append(raw_response)
                all_prompts_per_image[img_idx].append(prompt)

                # Parse this chunk's predictions
                chunk_predictions = response_scheme.parse_response(raw_response, chunk_attrs)
                all_predictions_per_image[img_idx].update(chunk_predictions)

        # Build final results
        results = []
        for img_idx in range(num_images):
            responses = all_responses_per_image[img_idx]
            prompts = all_prompts_per_image[img_idx]

            combined_response = "\n---CHUNK---\n".join(responses)
            combined_prompt = "\n---CHUNK---\n".join(prompts)

            results.append(
                VLMPredictionResult(
                    predictions=all_predictions_per_image[img_idx],
                    raw_response=combined_response,
                    prompt=combined_prompt,
                    chunk_responses=responses if len(chunks) > 1 else None,
                    chunk_prompts=prompts if len(chunks) > 1 else None,
                )
            )

        return results
