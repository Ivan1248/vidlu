"""
Base class and utilities for zero-shot VLM road attribute predictors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ..image_utils import to_pil_image as _to_pil_image  # noqa: F401
from ..prompts import DEFAULT_DETAIL_LEVEL, DetailLevel
from ..response_scheme import ResponseScheme, make_response_scheme
from ..response_parser import AttributePrediction


@dataclass
class VLMPredictionResult:
    """Result of a VLM prediction for a single image."""

    predictions: dict[str, AttributePrediction]
    raw_response: str
    prompt: str
    # For chunked predictions, store all chunks
    chunk_responses: list[str] | None = None
    chunk_prompts: list[str] | None = None
    # Thinking/reasoning text per chunk (None when thinking not enabled or absent)
    thinking_texts: list[str | None] | None = None


class BaseVLMPredictor(ABC):
    """Abstract base class for zero-shot road attribute classification.

    Handles shared logic for prompt building, attribute chunking, and result
    merging. Subclasses must implement ``_load_model()`` and
    ``_generate_single()``.

    Args:
        model_id: HuggingFace model ID.
        max_response_tokens: Maximum tokens to generate per inference call.
        response_scheme: ResponseScheme that controls how prompts are built and
            responses parsed. When None, a StandardResponseScheme is created
            lazily from the attribute metadata passed to ``predict()``.
        prompt_config_path: Optional YAML path for PromptBuilder configuration.
            Only used when ``response_scheme`` is None (to build the default
            StandardResponseScheme).
        chunk_size: Max attributes per VLM call (for long attribute lists).
        min_new_tokens: Minimum tokens to generate.
        debug: Enable debug output.
    """

    def __init__(
        self,
        model_id: str,
        max_response_tokens: int = 512,
        response_scheme: ResponseScheme | None = None,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 10,
        min_new_tokens: int = 0,
        debug: bool = False,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.max_response_tokens = max_response_tokens
        self.prompt_config_path = Path(prompt_config_path) if prompt_config_path else None
        self.chunk_size = chunk_size
        self.min_new_tokens = int(min_new_tokens)
        self.debug = debug
        self.enable_thinking = enable_thinking
        self.temperature = temperature

        self._response_scheme: ResponseScheme | None = response_scheme

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and processor/engine."""
        pass

    @abstractmethod
    def _generate_single(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> tuple[str, str | None]:
        """Generate response for a single prompt+image.

        Returns:
            Tuple of (clean_response, thinking_text). thinking_text is None when
            thinking is not enabled or no thinking block was found.
        """
        pass

    def _get_response_scheme(
        self,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
    ) -> ResponseScheme:
        """Return the response scheme, constructing a default if not provided."""
        if self._response_scheme is not None:
            return self._response_scheme

        self._response_scheme = make_response_scheme(
            "standard", attr_to_value_to_class_idx, self.prompt_config_path
        )
        return self._response_scheme

    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attrs_to_include: Sequence[str] | None = None,
        *,
        detail_level: DetailLevel = DEFAULT_DETAIL_LEVEL,
        custom_prompt: str | None = None,
    ) -> VLMPredictionResult:
        """Predict road attributes for a single image.

        Args:
            image: Input image in any supported format.
            attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
            attrs_to_include: Subset of attributes to classify.
            detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
            custom_prompt: Override the generated prompt entirely (skips chunking).

        Returns:
            VLMPredictionResult containing predictions and raw response.
        """
        self._load_model()
        pil_image = _to_pil_image(image)

        if attrs_to_include is None:
            attrs_to_include = list(attr_to_value_to_class_idx.keys())

        response_scheme = self._get_response_scheme(attr_to_value_to_class_idx)

        # If custom prompt provided, use it directly (no chunking, raw parse)
        if custom_prompt is not None:
            raw_response, thinking_text = self._generate_single(pil_image, custom_prompt)
            predictions = response_scheme.parse_response(raw_response, attrs_to_include)
            return VLMPredictionResult(
                predictions=predictions,
                raw_response=raw_response,
                prompt=custom_prompt,
                thinking_texts=[thinking_text] if thinking_text is not None else None,
            )

        # Split attributes into chunks to avoid context overflow
        attrs_list = list(attrs_to_include)
        chunks = [attrs_list[i : i + self.chunk_size] for i in range(0, len(attrs_list), self.chunk_size)]

        if self.debug:
            print(f"[DEBUG] Splitting {len(attrs_list)} attributes into {len(chunks)} chunks")

        all_predictions: dict[str, AttributePrediction] = {}
        all_prompts: list[str] = []
        all_responses: list[str] = []
        all_thinking_texts: list[str | None] = []

        for chunk_idx, chunk_attrs in enumerate(chunks):
            if self.debug:
                print(f"[DEBUG] Processing chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_attrs)} attrs")

            prompt = response_scheme.build_prompt(chunk_attrs, detail_level=detail_level)
            all_prompts.append(prompt)

            raw_response, thinking_text = self._generate_single(pil_image, prompt)
            all_responses.append(raw_response)
            all_thinking_texts.append(thinking_text)

            chunk_predictions = response_scheme.parse_response(raw_response, chunk_attrs)
            all_predictions.update(chunk_predictions)

        combined_response = "\n---CHUNK---\n".join(all_responses)
        combined_prompt = "\n---CHUNK---\n".join(all_prompts)
        has_thinking = any(t is not None for t in all_thinking_texts)

        return VLMPredictionResult(
            predictions=all_predictions,
            raw_response=combined_response,
            prompt=combined_prompt,
            chunk_responses=all_responses if len(chunks) > 1 else None,
            chunk_prompts=all_prompts if len(chunks) > 1 else None,
            thinking_texts=all_thinking_texts if has_thinking else None,
        )

    def predict_batch(
        self,
        images: Sequence[Image.Image | np.ndarray | torch.Tensor],
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attrs_to_include: Sequence[str] | None = None,
        *,
        detail_level: DetailLevel = DEFAULT_DETAIL_LEVEL,
    ) -> list[VLMPredictionResult]:
        """Predict road attributes for multiple images.

        Args:
            images: Sequence of input images.
            attr_to_value_to_class_idx: Mapping of attribute name -> {value -> class_idx}.
            attrs_to_include: Subset of attributes to classify.
            detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").

        Returns:
            List of VLMPredictionResult, one per image.
        """
        results = []
        for image in images:
            result = self.predict(
                image,
                attr_to_value_to_class_idx,
                attrs_to_include,
                detail_level=detail_level,
            )
            results.append(result)
        return results
