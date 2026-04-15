"""
Base class and utilities for zero-shot VLM road attribute predictors.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .prompts import DEFAULT_DETAIL_LEVEL, DetailLevel
from .response_scheme import ResponseScheme, make_response_scheme
from .response_parser import AttributePrediction

# Regex patterns for stripping thinking blocks from model responses.
# Qwen3: <think>...</think>
_QWEN_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Gemma 4: <|channel>thought\n...\n<channel|>
_GEMMA_THINK_RE = re.compile(r"<\|channel>thought\n.*?\n<channel\|>", re.DOTALL)


def strip_thinking(raw_response: str) -> tuple[str, str | None]:
    """Remove thinking/reasoning blocks from a VLM response.

    Handles both Qwen3 (``<think>...</think>``) and Gemma 4
    (``<|channel>thought\\n...\\n<channel|>``) formats.

    Returns:
        Tuple of (clean_response, thinking_text).  *thinking_text* is None
        when no thinking block was found.
    """
    for pattern in (_QWEN_THINK_RE, _GEMMA_THINK_RE):
        match = pattern.search(raw_response)
        if match:
            thinking_text = match.group(0)
            clean = raw_response[:match.start()] + raw_response[match.end():]
            return clean.strip(), thinking_text
    return raw_response, None


@dataclass
class VLMPredictionResult:
    """Result of a VLM prediction for a single image."""

    predictions: dict[str, AttributePrediction]
    raw_response: str
    prompt: str
    # For chunked predictions, store all chunks
    chunk_responses: list[str] | None = None
    chunk_prompts: list[str] | None = None


def _to_pil_image(
    image: Image.Image | np.ndarray | torch.Tensor,
) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        # Assume (C, H, W) or (H, W, C) format
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4):  # (C, H, W)
                image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
        elif image.ndim == 2:
            image = image.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")

    if isinstance(image, np.ndarray):
        # Handle float [0, 1] -> uint8 [0, 255]
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return Image.fromarray(image)

    raise TypeError(f"Unsupported image type: {type(image)}")


def extract_center_frame(
    rgb_sequence: torch.Tensor,
    frame_index: int = 0,
) -> Image.Image:
    """Extract a single frame from an RGB sequence tensor.

    The dataset returns sequences shaped (S, C, H, W) where S is sequence length.
    This function extracts frame at frame_index and converts to PIL Image.

    Args:
        rgb_sequence: Tensor shaped (S, C, H, W) with values in [0, 1].
        frame_index: Which frame to extract (0 = current segment).

    Returns:
        PIL Image of the extracted frame.
    """
    if rgb_sequence.ndim != 4:
        raise ValueError(f"Expected 4D tensor (S, C, H, W), got shape {rgb_sequence.shape}")

    frame = rgb_sequence[frame_index]  # (C, H, W)
    return _to_pil_image(frame)


class BaseVLMPredictor(ABC):
    """Abstract base class for zero-shot road attribute classification.

    Handles shared logic for prompt building, attribute chunking, and result
    merging. Subclasses must implement ``_load_model()`` and
    ``_generate_single()``.

    Args:
        model_id: HuggingFace model ID.
        max_new_tokens: Maximum tokens to generate per inference call.
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
        max_new_tokens: int = 512,
        response_scheme: ResponseScheme | None = None,
        prompt_config_path: str | Path | None = None,
        chunk_size: int = 10,
        min_new_tokens: int = 0,
        debug: bool = False,
        enable_thinking: bool = False,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
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
    ) -> str:
        """Generate response for a single prompt+image."""
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
            raw_response = self._generate_single(pil_image, custom_prompt)
            predictions = response_scheme.parse_response(raw_response, attrs_to_include)
            return VLMPredictionResult(
                predictions=predictions,
                raw_response=raw_response,
                prompt=custom_prompt,
            )

        # Split attributes into chunks to avoid context overflow
        attrs_list = list(attrs_to_include)
        chunks = [attrs_list[i : i + self.chunk_size] for i in range(0, len(attrs_list), self.chunk_size)]

        if self.debug:
            print(f"[DEBUG] Splitting {len(attrs_list)} attributes into {len(chunks)} chunks")

        all_predictions: dict[str, AttributePrediction] = {}
        all_prompts: list[str] = []
        all_responses: list[str] = []

        for chunk_idx, chunk_attrs in enumerate(chunks):
            if self.debug:
                print(f"[DEBUG] Processing chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_attrs)} attrs")

            prompt = response_scheme.build_prompt(chunk_attrs, detail_level=detail_level)
            all_prompts.append(prompt)

            raw_response = self._generate_single(pil_image, prompt)
            all_responses.append(raw_response)

            chunk_predictions = response_scheme.parse_response(raw_response, chunk_attrs)
            all_predictions.update(chunk_predictions)

        combined_response = "\n---CHUNK---\n".join(all_responses)
        combined_prompt = "\n---CHUNK---\n".join(all_prompts)

        return VLMPredictionResult(
            predictions=all_predictions,
            raw_response=combined_response,
            prompt=combined_prompt,
            chunk_responses=all_responses if len(chunks) > 1 else None,
            chunk_prompts=all_prompts if len(chunks) > 1 else None,
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
