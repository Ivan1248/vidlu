"""
Base class and utilities for VLM road attribute predictors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ..image_utils import to_pil_image as _to_pil_image  # noqa: F401
from ..prompts import DEFAULT_DETAIL_LEVEL, DetailLevel
from ..response_scheme import (DEFAULT_RESPONSE_TOKEN_MARGIN, ResponseScheme, make_response_scheme)
from ..response_parser import AttributePrediction
from .generation import get_text_tokenizer
from .thinking import DEFAULT_THINKING_BUDGET, split_thinking

# Sentinel for "put every attribute in one session".
ALL_ATTRS_IN_ONE_SESSION = None


def attribute_sessions(
    attrs_to_include: Sequence[str],
    attrs_per_session: int | None,
) -> list[list[str]]:
    """Splits the attributes into the groups that each get their own session.

    A *session* is one independent (image, prompt) -> response exchange. This is
    the single knob the per-attribute evaluation turns: ``None`` asks for every
    attribute in one prompt (what fine-tuning trains against), ``1`` gives each
    attribute its own prompt and its own session.

    Args:
        attrs_to_include: Ordered attribute names to classify.
        attrs_per_session: Maximum attributes per session, or None for all of
            them in one.

    Returns:
        A list of attribute groups covering ``attrs_to_include`` in order.
    """
    attrs = list(attrs_to_include)
    if attrs_per_session is None:
        return [attrs]
    if attrs_per_session < 1:
        raise ValueError(
            f"attrs_per_session must be at least 1, or None for all attributes in one"
            f" session; got {attrs_per_session}.")
    return [attrs[i:i + attrs_per_session] for i in range(0, len(attrs), attrs_per_session)]


@dataclass
class VLMSessionResult:
    """One (image, prompt) -> response exchange, and what was parsed out of it.

    Kept per session rather than concatenated because per-attribute evaluation
    produces one of these per attribute: with 41 of them, a single joined blob
    can no longer tell you which response belonged to which question, which is
    exactly what the analysis needs.
    """

    attrs: list[str]
    prompt: str
    response: str
    predictions: dict[str, AttributePrediction]
    thinking: str | None = None
    is_truncated: bool | None = None
    num_response_tokens: int | None = None


@dataclass
class VLMPredictionResult:
    """Result of a VLM prediction for a single image, over all its sessions."""

    predictions: dict[str, AttributePrediction]
    sessions: list[VLMSessionResult] = field(default_factory=list)

    @property
    def responses(self) -> list[str]:
        return [s.response for s in self.sessions]

    @property
    def raw_response(self) -> str:
        """All session responses, for display. Prefer ``sessions`` when parsing."""
        return "\n---SESSION---\n".join(self.responses)


class BaseVLMPredictor(ABC):
    """Abstract base class for road attribute classification by generation.

    Handles shared logic for prompt building, splitting the attributes into
    sessions, and merging the results. Subclasses must implement
    ``_load_model()`` and ``_generate_batch()``.

    Args:
        model_id: HuggingFace model ID.
        max_response_tokens: Maximum tokens to generate per session. None derives
            a bound per session from the response scheme, which knows the format
            and the closed value set (see ``ResponseScheme.compute_max_response_tokens``).
        response_scheme: ResponseScheme that controls how prompts are built and
            responses parsed. When None, a StandardResponseScheme is created
            lazily from the attribute metadata passed to ``predict()``.
        prompt_config_path: Optional YAML path for PromptBuilder configuration.
            Only used when ``response_scheme`` is None (to build the default
            StandardResponseScheme).
        attrs_per_session: Attributes per VLM session; None puts them all in one.
        response_token_margin: Absolute tokens added to a derived
            ``max_response_tokens``. See ``ResponseScheme.compute_max_response_tokens``.
        min_new_tokens: Minimum tokens to generate.
        thinking_budget: Tokens allowed for reasoning on top of the response budget,
            when ``enable_thinking``. See ``single_call_budget``.
        debug: Enable debug output.
    """

    def __init__(
        self,
        model_id: str,
        max_response_tokens: int | None = 512,
        response_scheme: ResponseScheme | None = None,
        prompt_config_path: str | Path | None = None,
        attrs_per_session: int | None = ALL_ATTRS_IN_ONE_SESSION,
        response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
        min_new_tokens: int = 0,
        debug: bool = False,
        enable_thinking: bool = False,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.max_response_tokens = max_response_tokens
        self.prompt_config_path = Path(prompt_config_path) if prompt_config_path else None
        self.attrs_per_session = attrs_per_session
        self.response_token_margin = int(response_token_margin)
        self.min_new_tokens = int(min_new_tokens)
        self.debug = debug
        self.enable_thinking = enable_thinking
        self.thinking_budget = int(thinking_budget)
        self.temperature = temperature

        self._response_scheme: ResponseScheme | None = response_scheme
        # Set by `_load_model` of the backend; some processors *are* the text tokenizer.
        self._processor = None
        # (prompt, response budget) per session, keyed by (attributes, detail level). The
        # scheme is fixed once set (see the `response_scheme` setter), so the entries cannot
        # go stale, and the budget derivation – which tokenizes every candidate value of
        # every attribute in the session – runs once per session instead of once per batch.
        self._session_prompt_and_budget: dict[tuple[tuple[str, ...], str], tuple[str, int]] = {}

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and processor/engine into ``self._processor`` etc."""
        pass

    @abstractmethod
    def _generate_batch(
        self,
        pil_images: Sequence[Image.Image],
        prompt: str,
        max_response_tokens: int,
    ) -> list[tuple[str, str | None, bool | None]]:
        """Responses ``prompt`` for each image, in one batch where the backend can.

        The prompt is shared across the batch on purpose: see
        ``_predict_sessions`` for why that is the axis to batch along.

        Returns:
            One (clean_response, thinking_text, is_truncated) tuple per image, in
            order. thinking_text is None when thinking is not enabled or no
            thinking block was found; is_truncated is None when the backend does
            not report it.
        """
        pass

    @property
    def tokenizer(self):
        """The text tokenizer, for measuring the response-token bound."""
        self._load_model()
        return get_text_tokenizer(self._processor)

    @property
    def response_scheme(self) -> ResponseScheme | None:
        """The prompt/response convention, or None until one is set or derived."""
        return self._response_scheme

    @response_scheme.setter
    def response_scheme(self, scheme: ResponseScheme) -> None:
        """Adopts a scheme, refusing one that disagrees with what is already set.

        Prompt building, ground-truth formatting and parsing are one convention;
        a predictor left to construct its own default would prompt in one format
        while the dataset scores in another. So evaluation hands the dataset's
        scheme over rather than hoping the two coincide, and a genuine mismatch
        is an error instead of a silently wrong run.
        """
        if self._response_scheme is not None and self._response_scheme is not scheme:
            raise ValueError(
                f"{type(self).__name__} already uses a "
                f"{type(self._response_scheme).__name__}, which is not the "
                f"{type(scheme).__name__} being assigned. The predictor and the dataset "
                f"have to share one convention; construct the predictor without a "
                f"response_scheme to let the dataset's be used.")
        self._response_scheme = scheme

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

    def session_response_budget(
        self,
        response_scheme: ResponseScheme,
        session_attrs: Sequence[str],
    ) -> int:
        """Response-token budget for one session.

        ``max_response_tokens=None`` derives it from the response scheme, which
        knows the format and the closed value set and can therefore compute a
        bound no compliant response can exceed – so it scales with the session
        instead of being a guess that is loose for one attribute and too tight
        for 41.
        """
        if self.max_response_tokens is not None:
            return self.max_response_tokens
        return response_scheme.compute_max_response_tokens(
            self.tokenizer, session_attrs, margin_tokens=self.response_token_margin)

    def _get_session_prompt_and_budget(
        self,
        response_scheme: ResponseScheme,
        session_attrs: Sequence[str],
        detail_level: DetailLevel,
    ) -> tuple[str, int]:
        """The prompt of one session and its response budget, computed once per session."""
        key = (tuple(session_attrs), detail_level)
        if key not in self._session_prompt_and_budget:
            self._session_prompt_and_budget[key] = (
                response_scheme.build_prompt(session_attrs, detail_level=detail_level),
                self.session_response_budget(response_scheme, session_attrs))
        return self._session_prompt_and_budget[key]

    def single_call_budget(self, response_budget: int) -> int:
        """Total new tokens for a backend that generates reasoning and response in one call.

        Reasoning goes first and shares the cap, so a budget sized for the response
        alone leaves nothing for it: the model reasons until the cap, emits no
        closing delimiter, and the reasoning gets parsed as the response. The
        allowance is added here rather than folded into ``max_response_tokens``
        so that it applies to a *derived* budget too, and so that
        ``session_response_budget`` keeps meaning response tokens – which is what
        ``VLMClassifierPredictor`` hands to the classifier, whose
        ``_generate_within_budget`` splits the two phases and must not be given a
        response budget inflated by reasoning room.
        """
        return response_budget + self.thinking_budget if self.enable_thinking else response_budget

    def split_thinking_and_truncation(
        self,
        raw_response: str,
        is_truncated: bool | None,
    ) -> tuple[str, str | None, bool | None]:
        """Separates reasoning from the response in one single-call backend's output.

        A missing closing delimiter means the response never came: reasoning ran into
        the shared cap, and what is left would otherwise be parsed as the response. That
        is reported as truncation, which the evaluation already surfaces, rather than
        scored as a wrong response.
        """
        if not self.enable_thinking:
            return raw_response, None, is_truncated
        response, thinking = split_thinking(raw_response)
        if self.debug and thinking:
            print(f"[DEBUG] Thinking ({len(thinking)} chars): {thinking[:200]}...")
        # `split_thinking` returns no thinking text exactly when no closing delimiter
        # was found. `is_truncated` is passed through otherwise, None included.
        return response, thinking, True if thinking is None else is_truncated

    def _predict_sessions(
        self,
        pil_images: Sequence[Image.Image],
        attr_to_value_to_class_idx: dict[str, dict[str, int]],
        attrs_to_include: Sequence[str],
        detail_level: DetailLevel,
    ) -> list[VLMPredictionResult]:
        """Runs every session for every image and merges the predictions.

        The loop is attribute-group-major, batching over images with the prompt
        held fixed. That axis is not arbitrary: the chat layout is
        text-then-image, so a fixed prompt is a shared prefix that the backend
        can cache, every row of the batch comes out the same length (no padding),
        and no image is encoded more than once per session. Batching over
        *attributes* instead would re-encode one image once per attribute and
        produce ragged rows.
        """
        response_scheme = self._get_response_scheme(attr_to_value_to_class_idx)
        sessions = attribute_sessions(attrs_to_include, self.attrs_per_session)

        if self.debug:
            print(f"[DEBUG] {len(pil_images)} image(s) x {len(sessions)} session(s)"
                  f" over {len(list(attrs_to_include))} attributes")

        results = [VLMPredictionResult(predictions={}) for _ in pil_images]
        for session_idx, session_attrs in enumerate(sessions):
            prompt, budget = self._get_session_prompt_and_budget(
                response_scheme, session_attrs, detail_level)
            if self.debug:
                print(f"[DEBUG] session {session_idx + 1}/{len(sessions)}:"
                      f" {len(session_attrs)} attr(s), budget {budget} tokens")

            for result, (response, thinking, is_truncated) in zip(
                    results, self._generate_batch(pil_images, prompt, budget)):
                predictions = response_scheme.parse_response(response, session_attrs)
                result.predictions.update(predictions)
                result.sessions.append(VLMSessionResult(
                    attrs=list(session_attrs),
                    prompt=prompt,
                    response=response,
                    predictions=predictions,
                    thinking=thinking,
                    is_truncated=is_truncated,
                    num_response_tokens=len(
                        self.tokenizer.encode(response, add_special_tokens=False)),
                ))
        return results

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
            custom_prompt: Override the generated prompt entirely (one session,
                asking for every attribute in ``attrs_to_include``).

        Returns:
            VLMPredictionResult containing predictions and the session records.
        """
        if custom_prompt is None:
            return self.predict_batch(
                [image], attr_to_value_to_class_idx, attrs_to_include,
                detail_level=detail_level)[0]

        self._load_model()
        pil_image = _to_pil_image(image)
        if attrs_to_include is None:
            attrs_to_include = list(attr_to_value_to_class_idx.keys())
        response_scheme = self._get_response_scheme(attr_to_value_to_class_idx)

        budget = self.session_response_budget(response_scheme, attrs_to_include)
        (response, thinking, is_truncated), = self._generate_batch(
            [pil_image], custom_prompt, budget)
        predictions = response_scheme.parse_response(response, attrs_to_include)
        return VLMPredictionResult(
            predictions=predictions,
            sessions=[VLMSessionResult(
                attrs=list(attrs_to_include), prompt=custom_prompt, response=response,
                predictions=predictions, thinking=thinking, is_truncated=is_truncated,
                num_response_tokens=len(
                    self.tokenizer.encode(response, add_special_tokens=False)))],
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
        self._load_model()
        if attrs_to_include is None:
            attrs_to_include = list(attr_to_value_to_class_idx.keys())
        return self._predict_sessions(
            [_to_pil_image(image) for image in images],
            attr_to_value_to_class_idx,
            attrs_to_include,
            detail_level,
        )
