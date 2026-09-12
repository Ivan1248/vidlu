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

from .base import BaseVLMPredictor
from ..response_scheme import DEFAULT_RESPONSE_TOKEN_MARGIN, ResponseScheme
from .thinking import DEFAULT_THINKING_BUDGET


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
        max_response_tokens: int | None = 512,
        response_scheme: "ResponseScheme | None" = None,
        prompt_config_path: str | Path | None = None,
        attrs_per_session: int | None = None,
        response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
        min_new_tokens: int = 0,
        debug: bool = False,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = 0.0,
    ):
        super().__init__(
            model_id=model_id,
            max_response_tokens=max_response_tokens,
            response_scheme=response_scheme,
            prompt_config_path=prompt_config_path,
            attrs_per_session=attrs_per_session,
            response_token_margin=response_token_margin,
            min_new_tokens=min_new_tokens,
            debug=debug,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = int(max_model_len)
        self.trust_remote_code = trust_remote_code

        self._llm = None  # loaded lazily, with `self._processor`

    def _load_model(self) -> None:
        """Load the vLLM engine and processor (called on first prediction)."""
        if self._llm is not None and self._processor is not None:
            return

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        print(f"[{type(self).__name__}] Loading model with vLLM: {self.model_id}")

        from vllm import LLM
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
            # The session loop holds the prompt fixed and varies the image, and
            # the chat layout is text-then-image, so the whole prompt is a shared
            # prefix across the batch – which is exactly what this reuses.
            enable_prefix_caching=True,
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        print(f"[{type(self).__name__}] Model loaded successfully")

    @abstractmethod
    def _prepare_vllm_input(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> dict:
        """Prepare input in vLLM format for the specific model family."""
        pass

    def _sampling_params(self, max_response_tokens: int):
        """Sampling parameters for one session.

        Built per session rather than once at load time because the response
        budget is derived from how many attributes the session asks about.
        """
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.temperature,
            max_tokens=max_response_tokens,
            min_tokens=self.min_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )

    def _generate_batch(
        self,
        pil_images: Sequence[Image.Image],
        prompt: str,
        max_response_tokens: int,
    ) -> list[tuple[str, str | None, bool | None]]:
        """Responses ``prompt`` for every image in one ``llm.generate`` call.

        ``max_response_tokens`` is the *response* budget; reasoning shares this
        backend's single cap, so ``single_call_budget`` adds its allowance.
        """
        budget = self.single_call_budget(max_response_tokens)
        if self.debug:
            print(f"[DEBUG] vLLM batch of {len(pil_images)}, prompt chars: {len(prompt)},"
                  f" budget {budget} tokens")

        vllm_inputs = [self._prepare_vllm_input(img, prompt) for img in pil_images]
        outputs = self._llm.generate(
            vllm_inputs, self._sampling_params(budget), use_tqdm=False)

        results = []
        for output in outputs:
            completion = output.outputs[0]
            raw_response = completion.text
            # vLLM reports why it stopped, so truncation needs no token
            # inspection: "length" means it ran into the cap.
            is_truncated = completion.finish_reason == "length"

            results.append(self.split_thinking_and_truncation(raw_response, is_truncated))
        return results
