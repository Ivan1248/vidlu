"""
Base class for HuggingFace transformers-backed VLM predictors.

Extracts shared model loading (env vars, flash-attention fallback, lazy init)
and generation logic (gen_kwargs, token slicing, thinking stripping) so that
model-specific subclasses only need to implement ``_load_hf_model()`` and
``_build_generation_inputs()``.
"""

import os
from abc import abstractmethod
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .base import BaseVLMPredictor
from ..response_scheme import DEFAULT_RESPONSE_TOKEN_MARGIN, ResponseScheme
from .generation import get_eos_token_ids, find_truncated_rows
from .thinking import DEFAULT_THINKING_BUDGET


class BaseHFPredictor(BaseVLMPredictor):
    """Abstract HF-transformers predictor with shared loading and generation.

    Subclasses must implement:
    - ``_load_hf_model()`` to load the model and processor into
      ``self._model`` and ``self._processor``.
    - ``_build_generation_inputs()`` to tokenize a prompt+image pair into
      a dict-like object suitable for ``model.generate(**inputs)``.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        max_response_tokens: int | None = 512,
        response_scheme: "ResponseScheme | None" = None,
        prompt_config_path: str | Path | None = None,
        attrs_per_session: int | None = None,
        response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
        min_new_tokens: int = 0,
        debug: bool = False,
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
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.use_flash_attention = use_flash_attention

        self._model = None  # loaded lazily, with `self._processor`

    def _load_model(self) -> None:
        """Loads the model and processor (called on first prediction)."""
        if self._model is not None and self._processor is not None:
            return

        # Defensive environment settings
        if "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        if "TRANSFORMERS_NO_TF" not in os.environ:
            os.environ["TRANSFORMERS_NO_TF"] = "1"

        class_name = type(self).__name__
        print(f"[{class_name}] Loading model: {self.model_id}")

        try:
            attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
            device_map = {"": self.device} if isinstance(self.device, (str, torch.device)) else "auto"
            self._load_hf_model(attn_impl, device_map)
        except Exception as e:
            self._model = None
            self._processor = None
            if self.use_flash_attention:
                raise RuntimeError(
                    "Model load failed with flash_attention_2 enabled. "
                    "Retry with use_flash_attention=False (CLI: --no-flash-attention). "
                    f"Original error: {e}"
                ) from e
            raise e

        print(f"[{class_name}] Model loaded successfully")

    @abstractmethod
    def _load_hf_model(self, attn_impl: str, device_map: dict | str) -> None:
        """Loads the HF model and processor into self._model and self._processor.

        Called inside ``_load_model()`` after env-var setup. The caller handles
        the flash-attention error fallback, so this method should just attempt
        the load and let exceptions propagate.

        Args:
            attn_impl: Attention implementation string (e.g. "flash_attention_2").
            device_map: Device map for ``from_pretrained``.
        """

    @abstractmethod
    def _build_generation_inputs(
        self,
        pil_images: Sequence[Image.Image],
        prompt: str,
    ) -> dict:
        """Prepares tokenized inputs for the specific model family.

        The same ``prompt`` is asked of every image, so the rows differ only in
        their image and (for a fixed image size) come out equal-length. Pad on
        the *left* regardless: ``generate`` continues from the end of the
        sequence, so right padding would put pad tokens where the response starts.

        Returns:
            A dict-like object (dict or BatchEncoding) that supports
            ``.to(device)``, has an ``input_ids`` key, and can be unpacked
            into ``model.generate(**inputs)``.
        """

    def _generate_batch(
        self,
        pil_images: Sequence[Image.Image],
        prompt: str,
        max_response_tokens: int,
    ) -> list[tuple[str, str | None, bool | None]]:
        """Responses ``prompt`` for every image in one ``generate`` call.

        ``max_response_tokens`` is the *response* budget; reasoning shares this
        backend's single cap, so ``single_call_budget`` adds its allowance. The
        two-phase alternative – reason under its own cap, then give the response
        its full budget – lives in ``_BaseVLMClassifier._generate_within_budget``,
        which owns its generation path.
        """
        budget = self.single_call_budget(max_response_tokens)
        inputs = self._build_generation_inputs(pil_images, prompt)
        inputs = inputs.to(self._model.device)

        if self.debug:
            print(f"[DEBUG] Input shape: {inputs['input_ids'].shape},"
                  f" Prompt chars: {len(prompt)}, budget {budget} tokens")

        with torch.no_grad():
            # HF's kwarg is max_new_tokens; passing the project's own name
            # through would trip generate()'s unused-model_kwargs validation.
            gen_kwargs = dict(max_new_tokens=budget)
            if self.min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = self.min_new_tokens
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Get only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_responses = self._processor.batch_decode(output_ids, skip_special_tokens=True)

        truncated = find_truncated_rows(output_ids, budget,
                                   get_eos_token_ids(self._model, self.tokenizer))

        results = [self.split_thinking_and_truncation(raw_response, is_truncated)
                   for raw_response, is_truncated in zip(raw_responses, truncated)]

        if self.debug:
            print(f"[DEBUG] Output tokens: {output_ids.shape[1]},"
                  f" first response: {results[0][0][:200]}...")

        return results
