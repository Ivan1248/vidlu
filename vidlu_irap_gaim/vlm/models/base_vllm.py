"""
Base class for vLLM-backed VLM predictors.

Extracts shared vLLM engine management, sampling parameters, and batched
inference logic so that model-specific subclasses only need to implement
``_prepare_vllm_input()``.
"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .base import (
    BaseVLMPredictor,
    VLMPredictionResult,
    _to_pil_image,
)
from .thinking import strip_thinking
from ..prompts import DEFAULT_DETAIL_LEVEL, DetailLevel


class BaseVLLMPredictor(BaseVLMPredictor):
    """Abstract vLLM predictor with shared engine setup and batched inference.

    Subclasses must implement ``_prepare_vllm_input()`` to convert a PIL image
    and prompt into the vLLM input dict for their specific model family.
    """

    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.80,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 8192,
        max_response_tokens: int = 512,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 15,
        min_new_tokens: int = 0,
        debug: bool = False,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_id=model_id,
            max_response_tokens=max_response_tokens,
            prompt_config_path=prompt_config_path,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = int(max_model_len)
        self.trust_remote_code = trust_remote_code

        # Lazy-loaded components
        self._llm = None
        self._processor = None
        self._sampling_params = None

    def _load_model(self) -> None:
        """Load the vLLM engine and processor (called on first prediction)."""
        if self._llm is not None and self._processor is not None:
            return

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        print(f"[{type(self).__name__}] Loading model with vLLM: {self.model_id}")

        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        tp_size = self.tensor_parallel_size
        if tp_size is None:
            tp_size = torch.cuda.device_count()
            print(f"[{type(self).__name__}] Using {tp_size} GPU(s) for tensor parallelism")

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            max_model_len=self.max_model_len,
            seed=0,
            enable_prefix_caching=True,
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_response_tokens,
            min_tokens=self.min_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )

        print(f"[{type(self).__name__}] Model loaded successfully")

    @abstractmethod
    def _prepare_vllm_input(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare input in vLLM format for the specific model family."""
        pass

    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> tuple[str, str | None]:
        """Generate response for a single prompt+image."""
        vllm_input = self._prepare_vllm_input(pil_image, prompt)

        if self.debug:
            print(f"[DEBUG] Prompt chars: {len(prompt)}")

        outputs = self._llm.generate([vllm_input], self._sampling_params, use_tqdm=False)
        raw_response = outputs[0].outputs[0].text

        thinking_text = None
        if self.enable_thinking:
            raw_response, thinking_text = strip_thinking(raw_response)
            if self.debug and thinking_text:
                print(f"[DEBUG] Thinking ({len(thinking_text)} chars): {thinking_text[:300]}...")

        if self.debug:
            print(f"[DEBUG] Response chars: {len(raw_response)}")
            print(f"[DEBUG] Response preview: {raw_response[:200]}...")

        return raw_response, thinking_text

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

        Args:
            images: Sequence of PIL images.
            attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
            attrs_to_include: Subset of attributes to classify.
            detail_level: Prompt detail level.

        Returns:
            List of VLMPredictionResult, one per image.
        """
        self._load_model()

        pil_images = [_to_pil_image(img) for img in images]
        num_images = len(pil_images)

        if attrs_to_include is None:
            attrs_to_include = list(attr_to_value_to_class_idx.keys())

        attrs_list = list(attrs_to_include)
        chunks = [attrs_list[i : i + self.chunk_size] for i in range(0, len(attrs_list), self.chunk_size)]

        if self.debug:
            print(f"[DEBUG] Batched inference: {num_images} images, {len(chunks)} attribute chunks")

        response_scheme = self._get_response_scheme(attr_to_value_to_class_idx)

        all_predictions_per_image: list[dict] = [{} for _ in range(num_images)]
        all_prompts_per_image: list[list[str]] = [[] for _ in range(num_images)]
        all_responses_per_image: list[list[str]] = [[] for _ in range(num_images)]
        all_thinking_texts_per_image: list[list[str | None]] = [[] for _ in range(num_images)]

        for chunk_idx, chunk_attrs in enumerate(chunks):
            if self.debug:
                print(
                    f"[DEBUG] Processing chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_attrs)} attrs, {num_images} images"
                )

            prompt = response_scheme.build_prompt(chunk_attrs, detail_level=detail_level)
            vllm_inputs = [self._prepare_vllm_input(img, prompt) for img in pil_images]
            outputs = self._llm.generate(vllm_inputs, self._sampling_params, use_tqdm=False)

            for img_idx, output in enumerate(outputs):
                raw_response = output.outputs[0].text

                thinking_text = None
                if self.enable_thinking:
                    raw_response, thinking_text = strip_thinking(raw_response)
                    if self.debug and thinking_text:
                        print(f"[DEBUG] Thinking img {img_idx} ({len(thinking_text)} chars): {thinking_text[:200]}...")

                all_responses_per_image[img_idx].append(raw_response)
                all_prompts_per_image[img_idx].append(prompt)
                all_thinking_texts_per_image[img_idx].append(thinking_text)

                chunk_predictions = response_scheme.parse_response(raw_response, chunk_attrs)
                all_predictions_per_image[img_idx].update(chunk_predictions)

        results = []
        for img_idx in range(num_images):
            responses = all_responses_per_image[img_idx]
            prompts = all_prompts_per_image[img_idx]
            thinking_texts = all_thinking_texts_per_image[img_idx]
            has_thinking = any(t is not None for t in thinking_texts)

            combined_response = "\n---CHUNK---\n".join(responses)
            combined_prompt = "\n---CHUNK---\n".join(prompts)

            results.append(
                VLMPredictionResult(
                    predictions=all_predictions_per_image[img_idx],
                    raw_response=combined_response,
                    prompt=combined_prompt,
                    chunk_responses=responses if len(chunks) > 1 else None,
                    chunk_prompts=prompts if len(chunks) > 1 else None,
                    thinking_texts=thinking_texts if has_thinking else None,
                )
            )

        return results
