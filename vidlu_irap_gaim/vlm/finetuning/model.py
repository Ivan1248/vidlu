"""VLM classifier wrappers (Qwen3-VL, Qwen3.5, Gemma 4) for Vidlu training integration.

Each classifier:
- Loads its base HuggingFace model eagerly in ``initialize()`` so parameters exist before optimizer creation.
- Attaches a PEFT LoRA adapter so only the adapter weights are trainable.
- Optionally uses 4-bit NF4 quantization for base weights (QLoRA).
- Saves an adapter-only ``state_dict``.
- Exposes ``build_messages``, ``tokenize_batch``, and ``generate_for_eval`` so ``VLMTrainStep`` and ``VLMEvalStep`` stay backend-agnostic.
"""

from typing import Any, Sequence
import os

import numpy as np
import torch
from PIL import Image
from torch import nn

from vidlu_irap_gaim.peft_utils import (check_lora_match, make_nf4_quantization_config,
                                        normalize_lora_target_modules)
from vidlu_irap_gaim.vlm.image_utils import to_pil_image as _to_pil_image
from vidlu_irap_gaim.vlm.models.gemma_utils import build_gemma_generation_inputs
from vidlu_irap_gaim.vlm.models.generation import (bf16_autocast, find_truncated_rows,
                                                   get_eos_token_ids, get_text_tokenizer)
from vidlu_irap_gaim.vlm.models.qwen_utils import build_qwen_generation_inputs
from vidlu_irap_gaim.vlm.models.thinking import (
    DEFAULT_THINKING_BUDGET,
    GEMMA_THINKING_END,
    QWEN_THINKING_END,
    has_thinking_end,
    split_thinking,
)


# ----- shared tokenization helpers (used by subclass tokenize_batch) ----------

def _create_labels_with_prompt_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_positions: list[int],
) -> torch.Tensor:
    """Clones input_ids and masks prompt and padding tokens to -100 for CE loss."""
    labels = input_ids.clone()
    for i, response_start in enumerate(response_start_positions):
        labels[i, :response_start] = -100
        labels[i, attention_mask[i] == 0] = -100
    return labels


def _compute_response_start_positions(
    attention_mask: torch.Tensor,
    response_texts: list[str],
    tokenizer,
) -> list[int]:
    """Computes the per-sample index in the padded sequence where the assistant response begins.

    Handles both left- and right-padded batches by locating the first attended
    token via ``argmax(attention_mask == 1)``.

    The boundary is measured from the end of the sequence rather than by
    tokenizing the prompt: the processor expands ``<|image_pad|>`` into one token
    per vision patch, so a text-only prompt tokenization omits image patch tokens.
    Measuring from the response avoids this expansion.

    This assumes response tokenization is context-invariant. For Qwen chat
    templates, the generation prompt ends in trailing newlines that the
    pre-tokenizer isolates (``\\s*[\\r\\n]+``), preventing BPE merges across the
    prompt-response boundary.
    """
    positions = []
    for mask, response_text in zip(attention_mask, response_texts):
        response_len = len(tokenizer.encode(response_text, add_special_tokens=False))
        full_nonpadded_len = int(mask.sum())
        prompt_only_len = full_nonpadded_len - response_len
        first_real_token = int((mask == 1).long().argmax())
        positions.append(first_real_token + prompt_only_len)
    return positions


# ----- base classifier --------------------------------------------------------

class _BaseVLMClassifier(nn.Module):
    """Common QLoRA + adapter-checkpointing wiring shared by VLM classifiers.

    Subclasses provide:
      - ``_DEFAULT_MODEL_ID``        : default HF model id.
      - ``_DEFAULT_LORA_TARGET_MODULES`` : default LoRA target sublayer names.
      - ``_build_hf_model(load_kwargs)``  : load the HF base model.
      - ``build_messages(image, prompt, response=None)`` : chat-format messages.
      - ``tokenize_batch(...)``          : batched (image, prompt, response) → tensors.
      - ``_build_generation_inputs(images, prompts)`` : batched (image, prompt) →
        processor tensors to generate from.  Generation itself is shared:
        ``generate_batch_for_eval`` / ``generate_for_eval`` live here.
    """

    _DEFAULT_MODEL_ID: str = ""
    # tuple[str, ...] → PEFT suffix-matches each entry against module names.
    # str            → PEFT treats it as a regex (re.fullmatch on full path).
    _DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] | str = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    # Order in which attention implementations are tried at load time; the
    # first that loads is used. This only catches load-time failures –
    # an implementation that loads but fails during the forward pass must be
    # excluded manually (e.g. Gemma 4 and Qwen3.5 have head_dim=256, which
    # causes FlashAttention failures during the forward pass).
    _ATTN_IMPL_PREFS: tuple[str, ...] = ("flash_attention_2", "sdpa", "eager")
    # NF4 quantization of the base weights.  True works for plain (non-MoE)
    # VLMs (Qwen3-VL); some MoE models (Gemma 4) need bf16 instead.
    _DEFAULT_LOAD_IN_4BIT: bool = True
    # HF device_map for from_pretrained.
    #   None   → ``{"": self._device or "cuda"}`` (single-device load).
    #   "auto" → HF places shards across visible GPUs (naive model parallel).
    _DEFAULT_DEVICE_MAP: dict | str | None = None
    # Whether to request a reasoning block from the chat template.
    #   None  → omit parameter; use checkpoint default.
    #   bool  → pass ``enable_thinking=<bool>`` to all apply_chat_template calls.
    # Reasoning models require False: otherwise the generation prompt leaves an
    # open `` thinking`` tag, causing the model to emit a stray closing `` response``
    # before the response and allowing reasoning to consume the token budget.
    # Matches the name and semantics of ``BaseVLMPredictor.enable_thinking``.
    _DEFAULT_ENABLE_THINKING: bool | None = None
    # Closing delimiter this model family emits to end a reasoning block.  Used
    # to *force* the block closed when the reasoning budget runs out, so it must
    # name one delimiter rather than matching any (which is what the read-side
    # ``split_thinking`` does).
    _THINKING_END_DELIMITER: str = QWEN_THINKING_END
    # Whether this family's `generate` stays correct on *left-padded* inputs,
    # which a batch needs only when its rows have unequal length.  Batched
    # evaluation groups by a fixed prompt and varies the image, so the rows are
    # normally equal-length and no padding is added at all; this flag gates only
    # the ragged case, where a family that cannot be left-padded falls back to
    # generating row by row.
    _SUPPORTS_LEFT_PADDED_GENERATION: bool = True

    def __init__(
        self,
        model_id: str | None = None,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] | str | None = None,
        use_lora: bool = True,
        load_in_4bit: bool | None = None,
        use_gradient_checkpointing: bool = True,
        device_map: dict | str | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        input_adapter=None,
    ):
        super().__init__()
        self.model_id = model_id or self._DEFAULT_MODEL_ID
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        targets = (
            lora_target_modules if lora_target_modules is not None
            else self._DEFAULT_LORA_TARGET_MODULES
        )
        # Preserve str (regex); normalize list/tuple to tuple (suffix-match).
        self.lora_target_modules = targets if isinstance(targets, str) else tuple(targets)
        # False leaves the pretrained model unwrapped, which is only useful for inference with
        # the original weights (`loading.load_base_classifier`): there is then nothing to train
        # and nothing for `state_dict` to save.
        self.use_lora = use_lora
        self.load_in_4bit = (
            self._DEFAULT_LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # device_map=None here means "use class default" (which may itself be
        # None ⇒ single-device fallback computed at load time).
        self.device_map = self._DEFAULT_DEVICE_MAP if device_map is None else device_map
        # Same convention: None ⇒ class default, which may itself be None
        # meaning "leave the kwarg off and use the checkpoint's own default".
        self.enable_thinking = (
            self._DEFAULT_ENABLE_THINKING if enable_thinking is None else enable_thinking
        )
        self.thinking_budget = thinking_budget
        # Accepted for Vidlu factory compatibility; VLM tokenized inputs must
        # not be transformed by image-space adapters.
        self.input_adapter = input_adapter

        # Placeholder so Vidlu can detect device before the real model loads.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    # --- to be implemented by subclasses --------------------------------------

    def _build_hf_model(self, load_kwargs: dict) -> nn.Module:
        raise NotImplementedError

    def build_messages(self, pil_image, prompt: str, response: str | None = None) -> list[dict]:
        raise NotImplementedError

    def tokenize_batch(
        self,
        images: torch.Tensor,
        prompts: list[str],
        responses: list[str],
        targets: torch.Tensor,
        max_length: int = 8192,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _build_generation_inputs(self, pil_images: list, prompts: list[str]) -> dict:
        """Processor inputs for generation, padded on the left when rows are ragged.

        Padded on the left because ``generate`` continues from the end of the
        sequence. Right padding – used during the teacher-forced training forward pass
        in ``tokenize_batch`` – would place pad tokens where the response must begin.
        A batch built with a fixed prompt and varying images has equal-length rows
        without padding; the padding side applies only to ragged batches
        (see ``_SUPPORTS_LEFT_PADDED_GENERATION``).

        Returns:
            Processor output dictionary, on the CPU; the caller moves it to the model.
        """
        raise NotImplementedError

    def generate_batch_for_eval(
        self,
        images: Sequence,
        prompts: Sequence[str],
        max_response_tokens: int | None = 512,
        amp: bool = True,
        min_new_tokens: int = 0,
    ) -> list[tuple[str, str | None, bool]]:
        """Generates one response per (image, prompt) pair in a single batch.

        Primary generation path: owns the chat template, vision preprocessing, and
        the reasoning-response budget split. Callers requiring variations
        (``VLMClassifierPredictor``) pass arguments rather than reimplementing it;
        ``generate_for_eval`` is the single-element case.

        Batching amortizes latency across attributes by holding the prompt fixed and
        varying the image. Because the chat layout places text before image, a fixed
        prompt shares a common prefix, produces equal-length rows without padding,
        and avoids encoding images multiple times.

        Falls back to sequential row-by-row generation in two cases:
        1. Reasoning is enabled – the budget split is per-sequence, see
           ``_generate_within_budget``.
        2. The batch has ragged rows (``not inputs["attention_mask"].all()``) on a model
           family that does not support left-padded generation.

        Args:
            images: Inputs accepted by ``vidlu_irap_gaim.vlm.image_utils.to_pil_image``.
            prompts: One prompt per image.
            max_response_tokens: Maximum response tokens; None allows generation until
                EOS or context window limit.
            amp: Whether to run generation under bfloat16 mixed precision.
            min_new_tokens: Lower bound on response tokens, guarding against an empty
                response. Reasoning is never subject to this limit – see
                ``_generate_within_budget``.

        Returns:
            One (response, thinking_text, is_truncated) tuple per input, in order.
            *thinking_text* is None when thinking is disabled or no reasoning block was
            produced. *is_truncated* is True when generation exhausted the
            ``max_response_tokens`` budget without emitting EOS.
        """
        images, prompts = list(images), list(prompts)
        if len(images) != len(prompts):
            raise ValueError(
                f"[{type(self).__name__}] Got {len(images)} images but {len(prompts)}"
                + " prompts; generation pairs them positionally.")
        if not images:
            return []

        def generate_one_by_one():
            return [self.generate_batch_for_eval(
                        [image], [prompt], max_response_tokens=max_response_tokens,
                        amp=amp, min_new_tokens=min_new_tokens)[0]
                    for image, prompt in zip(images, prompts)]

        if len(images) > 1 and self.enable_thinking:
            return generate_one_by_one()

        pil_images = [_to_pil_image(image) for image in images]
        inputs = self._build_generation_inputs(pil_images, prompts)
        if (len(images) > 1 and not self._SUPPORTS_LEFT_PADDED_GENERATION
                and not bool(inputs["attention_mask"].all())):
            return generate_one_by_one()
        inputs = inputs.to(next(self._model.parameters()).device)

        raw_texts, is_truncated = self._generate_within_budget(
            inputs, max_response_tokens, amp, min_new_tokens=min_new_tokens)
        return [(*self._split_thinking(raw), truncated)
                for raw, truncated in zip(raw_texts, is_truncated)]

    def generate_for_eval(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        prompt: str,
        max_response_tokens: int | None = 512,
        amp: bool = True,
        min_new_tokens: int = 0,
    ) -> tuple[str, str | None, bool]:
        """Generates one response; the one-element case of ``generate_batch_for_eval``."""
        return self.generate_batch_for_eval(
            [image], [prompt], max_response_tokens=max_response_tokens, amp=amp,
            min_new_tokens=min_new_tokens)[0]

    @property
    def tokenizer(self):
        """The text tokenizer, for callers that need to measure token lengths.

        Some processors *are* the tokenizer rather than wrapping one.
        """
        return get_text_tokenizer(self._processor)

    def _split_thinking(self, raw_response: str) -> tuple[str, str | None]:
        """Separate reasoning from the response, if this model is reasoning."""
        if not self.enable_thinking:
            return raw_response, None
        return split_thinking(raw_response)

    def _eos_token_ids(self) -> set[int]:
        return get_eos_token_ids(self._model, self.tokenizer)

    def _truncated_rows(self, generated_ids: torch.Tensor, budget: int) -> list[bool]:
        """Per row: whether generation was cut off by the cap; see `find_truncated_rows`."""
        return find_truncated_rows(generated_ids, budget, self._eos_token_ids())

    def _context_length(self) -> int:
        """The model's context window, in tokens."""
        config = self._model.config
        # Composite (text + vision) configs keep the text limit in `text_config`.
        for holder in (getattr(config, "text_config", None), config):
            if (length := getattr(holder, "max_position_embeddings", None)) is not None:
                return length
        raise RuntimeError(
            f"[{type(self).__name__}] {self.model_id} does not declare"
            + " `max_position_embeddings`, so the context window is unknown. Pass an explicit"
            + " response budget instead of None.")

    def _resolve_response_budget(self, response_budget: int | None, num_prefix_tokens: int) -> int:
        """The response budget to generate under, resolving None to the rest of the context window.

        Args:
            response_budget: Maximum number of response tokens, or None for "no limit beyond the
                context window", in which case generation ends only at EOS or at that window.
            num_prefix_tokens: Length of what the response continues, which the window is shared
                with – the prompt, plus the reasoning block when there is one.
        """
        if response_budget is not None:
            return response_budget
        context_length = self._context_length()
        if (remaining := context_length - num_prefix_tokens) <= 0:
            raise ValueError(f"The {num_prefix_tokens} tokens generation would continue from"
                             + f" already fill the {context_length}-token context window.")
        return remaining

    def _generate_within_budget(
        self,
        inputs,
        response_budget: int | None,
        amp: bool,
        min_new_tokens: int = 0,
    ) -> tuple[list[str], list[bool]]:
        """Generates one response per row, returning ``(raw_texts, is_truncated)``.

        Without thinking, runs a single ``generate`` call capped at *response_budget*
        over the batch.

        With thinking enabled, reasoning and response would otherwise share a single token
        budget. Because reasoning precedes the response, a single call can exhaust the
        budget on reasoning alone and return no response. If no closing delimiter is emitted,
        the reasoning text would also be misparsed as the response. To prevent this,
        reasoning runs under ``thinking_budget``; if that cap is reached, the closing
        delimiter is forced and the response is generated under a separate ``response_budget``.
        Exhausting reasoning budget limits reasoning depth but guarantees a response.

        The continuation runs a second ``generate`` pass over the extended prefix,
        incurring one additional prefill instead of reusing the KV cache. This deliberate
        simplicity trade-off maintains backend agnosticism and applies only to samples
        that exhaust the reasoning budget.

        *min_new_tokens* prevents empty responses and applies only to the response call,
        never to reasoning – brief reasoning followed by a response remains valid.

        The reasoning path processes inputs row by row because the two-phase budget split
        forces thinking delimiters individually and continuation prefixes differ across
        rows. ``generate_batch_for_eval`` loops sequentially when thinking is enabled,
        passing batches of size one.
        """
        autocast = bf16_autocast(amp)
        processor = self._processor
        input_len = inputs["input_ids"].shape[1]
        response_kwargs = {"min_new_tokens": min_new_tokens} if min_new_tokens > 0 else {}

        def decode(ids) -> list[str]:
            return processor.batch_decode(ids, skip_special_tokens=True)

        if not self.enable_thinking:
            budget = self._resolve_response_budget(response_budget, input_len)
            with torch.no_grad(), autocast:
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=budget, **response_kwargs
                )
            generated_ids = output_ids[:, input_len:]
            return decode(generated_ids), self._truncated_rows(generated_ids, budget)

        if inputs["input_ids"].shape[0] != 1:
            raise RuntimeError(
                f"[{type(self).__name__}] Reasoning generation is single-row (see the"
                + " docstring), but got a batch of"
                + f" {inputs['input_ids'].shape[0]}.")

        def decode_one(ids) -> str:
            return decode(ids)[0]

        with torch.no_grad(), autocast:
            reasoning_output_ids = self._model.generate(
                **inputs, max_new_tokens=self.thinking_budget)
        reasoning_ids = reasoning_output_ids[:, input_len:]
        # Stopped on its own ⇒ the response is already complete.
        if not self._truncated_rows(reasoning_ids, self.thinking_budget)[0]:
            return decode(reasoning_ids), [False]

        prefix_ids = reasoning_output_ids
        if not has_thinking_end(decode_one(reasoning_ids)):
            forced_end_ids = self.tokenizer.encode(
                self._THINKING_END_DELIMITER, add_special_tokens=False
            )
            prefix_ids = torch.cat(
                [reasoning_output_ids,
                 torch.tensor([forced_end_ids], device=reasoning_output_ids.device,
                              dtype=reasoning_output_ids.dtype)], dim=1
            )

        continuation_inputs = dict(inputs)
        continuation_inputs["input_ids"] = prefix_ids
        # Batch size is 1 here, so there is no padding to mask out.
        continuation_inputs["attention_mask"] = torch.ones_like(prefix_ids)
        # Resolved against the extended prefix: the reasoning block shares the context window.
        budget = self._resolve_response_budget(response_budget, prefix_ids.shape[1])
        with torch.no_grad(), autocast:
            response_output_ids = self._model.generate(
                **continuation_inputs, max_new_tokens=budget, **response_kwargs
            )
        response_ids = response_output_ids[:, prefix_ids.shape[1]:]
        full_ids = torch.cat([prefix_ids[:, input_len:], response_ids], dim=1)
        return decode(full_ids), self._truncated_rows(response_ids, budget)

    # --- shared tokenize helpers ----------------------------------------------

    def _chat_template_kwargs(self) -> dict:
        """Jinja variables for ``apply_chat_template``; the one place that decides.

        Templates ignore variables they don't declare, so passing
        ``enable_thinking`` to a non-reasoning template (Qwen3-VL-Instruct,
        Gemma 4) is a no-op.  ``None`` omits it entirely.
        """
        return {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}

    def _render_chat_texts(
        self,
        messages_full_batch: list[list[dict]],
        messages_prompt_only_template: list[dict],
    ) -> tuple[list[str], list[str]]:
        """Renders full and (broadcast) prompt-only chat texts for the batch.

        VLMIrapDataset uses an identical prompt for every sample, so the
        prompt-only template text is computed once and broadcast across the
        batch (measurably faster than per-sample apply). Under
        ``VLM_VERIFY_PROMPT_ONLY=1`` the broadcast assumption is checked.

        Both renders get the same template kwargs so that the training prompt,
        the label boundary and the generation prompt cannot drift apart.
        """
        processor = self._processor
        template_kwargs = self._chat_template_kwargs()
        texts_full = [
            processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False, **template_kwargs
            )
            for m in messages_full_batch
        ]
        text_prompt_only = processor.apply_chat_template(
            messages_prompt_only_template,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        texts_prompt_only = [text_prompt_only] * len(messages_full_batch)
        if os.environ.get("VLM_VERIFY_PROMPT_ONLY") == "1":
            per_sample = [
                processor.apply_chat_template(
                    m[:-1], tokenize=False, add_generation_prompt=True, **template_kwargs
                )
                for m in messages_full_batch
            ]
            assert len(set(per_sample)) == 1 and per_sample[0] == text_prompt_only, (
                "prompt-only template output differs across samples"
            )
        # The label boundary is derived by slicing the prompt off the front of
        # the full render, so the prefix relation has to actually hold.  It is a
        # property of the chat template, not of this code: Qwen3.5's template
        # emits `<|im_start|>assistant\n<think>\\n` in *both* the
        # add_generation_prompt branch and the final-assistant-message branch,
        # which is what makes it line up.  A template change that broke this
        # would otherwise silently mask the wrong tokens.
        for text_full in texts_full:
            if not text_full.startswith(text_prompt_only):
                raise RuntimeError(
                    f"[{type(self).__name__}] Chat template broke the prompt-prefix "
                    f"assumption: the full render does not start with the prompt-only "
                    f"render, so the response boundary cannot be located.\n"
                    f"prompt-only: {text_prompt_only!r}\n"
                    f"full:        {text_full!r}"
                )
        return texts_full, texts_prompt_only

    def _assemble_tokenized_batch(
        self,
        inputs: dict[str, torch.Tensor],
        texts_full: list[str],
        texts_prompt_only: list[str],
        targets: torch.Tensor,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        """Computes labels and forwards every tensor produced by the processor."""
        # Safe because _render_chat_texts has verified the prefix relation.
        response_texts = [
            full[len(prompt_only):]
            for full, prompt_only in zip(texts_full, texts_prompt_only)
        ]
        response_start_positions = _compute_response_start_positions(
            attention_mask=inputs["attention_mask"],
            response_texts=response_texts,
            tokenizer=self._processor.tokenizer,
        )
        labels = _create_labels_with_prompt_mask(
            inputs["input_ids"], inputs["attention_mask"], response_start_positions
        )
        if (labels != -100).sum().item() == 0:
            raise RuntimeError(
                f"All labels are masked (-100): the response was completely "
                f"truncated. seq_len={inputs['input_ids'].shape[1]}, "
                f"max_length={max_length}, "
                f"response_start_positions={response_start_positions}."
            )
        result = {"labels": labels, "target": targets}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value
        return result

    # --- shared load / lifecycle ----------------------------------------------

    def initialize(self, init_input):
        self._load()

    def make_load_kwargs(self, *, device_map) -> dict:
        """Keyword arguments for the ``from_pretrained`` call in `_build_hf_model`.

        The single place that decides them, so that a diagnostic loading the same model
        (`vidlu_irap_gaim.tools.dump_module_names`) cannot drift from what training loads.
        ``attn_implementation`` is left to the caller: `_load` tries `_ATTN_IMPL_PREFS` in turn.

        Args:
            device_map: HF ``device_map`` for the load.
        """
        return dict(
            quantization_config=make_nf4_quantization_config() if self.load_in_4bit else None,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    def _load(self):
        from transformers import AutoProcessor

        if self._loaded:
            return

        device = self._device or "cuda"

        if "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        cls_name = type(self).__name__
        print(f"[{cls_name}] Loading {self.model_id}...")

        load_kwargs = self.make_load_kwargs(
            device_map=self.device_map if self.device_map is not None else {"": device})
        for attn_impl in self._ATTN_IMPL_PREFS:
            load_kwargs["attn_implementation"] = attn_impl
            try:
                self._model = self._build_hf_model(load_kwargs)
                if attn_impl != "eager":
                    print(f"[{cls_name}] Using attention: {attn_impl}")
                break
            except Exception:
                if attn_impl == "eager":
                    raise
                load_kwargs.pop("attn_implementation", None)

        if self.use_lora:
            # Imported here rather than at the top of the method so that inference with the
            # pretrained weights does not require PEFT to be installed.
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=normalize_lora_target_modules(self.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, peft_config)
            check_lora_match(self._model, self.lora_target_modules, cls_name)

        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        if self.use_gradient_checkpointing:
            # No need to clear config.use_cache: in transformers v5 forward()
            # defaults use_cache to None and never reads it off the config, and
            # on a composite (text+vision) config the flag lives on text_config
            # anyway, so setting it on the outer config was a no-op.
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # Silences a per-sample "Setting pad_token_id to eos_token_id" warning from
        # generate(). eos is the value HF falls back to, so outputs are unaffected.
        gen_cfg = getattr(self._model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            tokenizer = self.tokenizer
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
            gen_cfg.pad_token_id = pad_id

        self._loaded = True

        total_params = sum(p.numel() for p in self._model.parameters())
        adapter = "LoRA adapter" if self.use_lora else "no adapter (pretrained weights)"
        print(
            f"[{cls_name}] Loaded with {adapter}. Trainable: {trainable_params:,} / "
            f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    # --- forward / generate ---------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Ensure initialize() was called "
                "(this happens automatically in Vidlu's build_and_init_model)."
            )
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )

    def generate(self, **kwargs):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return self._model.generate(**kwargs)

    def to(self, device, *args, **kwargs):
        # Once a sharded model is loaded (device_map != None), a subsequent
        # .to(device) would try to move every parameter to a single device
        # and obliterate the sharding.  Treat such calls as a no-op so the
        # vidlu Trainer's later ``model.to(device)`` is harmless.
        if self._loaded and self.device_map is not None:
            return self
        self._device = device
        return super().to(device, *args, **kwargs)

    # --- adapter-only checkpointing -------------------------------------------

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Returns only LoRA adapter weights + config metadata.

        Reduces checkpoint size from ~10s of GB to ~100s of MB.
        """
        if not self._loaded:
            return {"_config": self._get_config_dict()}

        adapter_state = {
            name: param.data.cpu()
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }
        return {
            "_config": self._get_config_dict(),
            "_adapter_state": adapter_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        if "_config" not in state_dict:
            raise ValueError("Invalid checkpoint: missing _config")
        ckpt_model_id = state_dict["_config"]["model_id"]
        if ckpt_model_id != self.model_id:
            raise ValueError(
                f"Model ID mismatch: checkpoint has '{ckpt_model_id}', "
                f"but model expects '{self.model_id}'"
            )
        if "_adapter_state" in state_dict:
            self._load()
            adapter_state = state_dict["_adapter_state"]
            model_state = self._model.state_dict()
            loaded_count = 0
            for name, param in adapter_state.items():
                if name in model_state:
                    model_state[name].copy_(param.to(model_state[name].device))
                    loaded_count += 1
                elif strict:
                    raise KeyError(f"Missing key in model state dict: {name}")
            print(
                f"[{type(self).__name__}] Loaded {loaded_count} adapter parameters from checkpoint"
            )

    def _get_config_dict(self) -> dict[str, Any]:
        """Everything needed to reconstruct this classifier from a checkpoint alone.

        `classifier_class` is not a constructor argument – it names the class to construct, so
        `vidlu_irap_gaim.vlm.finetuning.loading` can load a checkpoint without being told which
        model family wrote it. Every other entry is a constructor keyword argument.
        """
        return {
            "classifier_class": type(self).__name__,
            "model_id": self.model_id,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
        }

    @property
    def processor(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return self._processor

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        return [p for p in self._model.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        if self._model is not None:
            self._model.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ----- Qwen3-VL ---------------------------------------------------------------

class Qwen3VLClassifier(_BaseVLMClassifier):
    """Qwen3-VL with LoRA, integrated into Vidlu training."""

    _DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    _DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

    def _build_hf_model(self, load_kwargs: dict) -> nn.Module:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)

    def build_messages(self, pil_image, prompt: str, response: str | None = None) -> list[dict]:
        from vidlu_irap_gaim.vlm.models.qwen_utils import build_qwen_chat_messages
        return build_qwen_chat_messages(pil_image, prompt, response=response)

    def tokenize_batch(
        self,
        images: torch.Tensor,
        prompts: list[str],
        responses: list[str],
        targets: torch.Tensor,
        max_length: int = 8192,
    ) -> dict[str, torch.Tensor]:
        """Qwen-VL tokenize: shared image processor + tokenizer call."""
        from qwen_vl_utils import process_vision_info

        batch_size = images.shape[0]
        messages_batch = [
            self.build_messages(_to_pil_image(images[i]), prompts[i], response=responses[i])
            for i in range(batch_size)
        ]
        all_image_inputs = []
        for m in messages_batch:
            img_inputs, _ = process_vision_info(m)
            all_image_inputs.extend(img_inputs if img_inputs else [])

        texts_full, texts_prompt_only = self._render_chat_texts(
            messages_batch, messages_batch[0][:-1]
        )
        inputs = self._processor(
            text=texts_full,
            images=all_image_inputs if all_image_inputs else None,
            return_tensors="pt",
            padding=True,
            # Pinned rather than inherited from the checkpoint's tokenizer
            # config.  Qwen3.5's Gated-DeltaNet layers zero the padded positions
            # and then run a causal conv + recurrence over the sequence, which
            # is only correct when the padding is on the right; left padding
            # would decay the recurrent state before the real tokens start.
            padding_side="right",
            truncation=True,
            max_length=max_length,
        )
        return self._assemble_tokenized_batch(
            inputs, texts_full, texts_prompt_only, targets, max_length
        )

    def _build_generation_inputs(self, pil_images: list, prompts: list[str]) -> dict:
        return build_qwen_generation_inputs(self._processor, pil_images, prompts,
                                            self._chat_template_kwargs())


# ----- Gemma 4 ----------------------------------------------------------------

class Gemma4VLClassifier(_BaseVLMClassifier):
    """Gemma 4 multimodal (MoE 26B-A4B by default) with LoRA + optional QLoRA.

    Notes for this model:
    - HF auto class is ``AutoModelForMultimodalLM`` (not Vision2Seq).
    - Image input is passed via ``processor.apply_chat_template(..., images=[pil])``;
      there is no qwen_vl_utils involvement.
    - 4-bit (NF4) quantization of the 128 MoE expert linears requires
      ``bitsandbytes >= 0.43``; if loading fails inside the experts, set
      ``load_in_4bit=False``.
    - Default LoRA targets are the standard attention proj names; if PEFT
      reports 0 trainable params, dump ``named_modules()`` and override
      ``lora_target_modules``.
    """

    _DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
    # SDPA is what Google/HF's official Gemma 4 fine-tuning recipe uses.
    # FlashAttention 2 loads but errors at forward with
    # ``FlashAttention forward only supports head dimension at most 256``
    # on many flash-attn builds – Gemma 4 text has head_dim=256 which is on
    # the boundary and trips a strict check.  Keep eager as the final fallback.
    _ATTN_IMPL_PREFS = ("sdpa", "eager")
    # bnb 4-bit silently skips this model's MoE expert linears (observed:
    # load consumes ~48 GB on one A6000, i.e. the experts stay bf16).  See
    # unslothai/unsloth#4907.  Default to bf16 and shard across visible GPUs
    # via ``device_map="auto"`` (naive model parallel on 4× A6000 fits
    # comfortably at ~12.5 GB/GPU for the base weights).
    _DEFAULT_LOAD_IN_4BIT = False
    _DEFAULT_DEVICE_MAP = "auto"
    _THINKING_END_DELIMITER = GEMMA_THINKING_END
    # Regex scoped to the language model only.
    #
    # Two reasons not to use the standard tuple-of-suffixes default:
    # 1. PEFT cannot wrap the vision tower's ``Gemma4ClippableLinear``
    #    (custom Linear wrapper) → "Target module ... is not supported".
    #    Restricting to the LLM is also the correct SFT default (vision
    #    tower stays frozen, matching huggingface-gemma-recipes' "llm-only").
    # 2. Including the MoE expert linears (``gate_proj``, ``up_proj``,
    #    ``down_proj``, ``gate_up_proj``) is required to give the adapter
    #    real capacity on this MoE.  Skipping them yields a suspiciously
    #    small trainable-param fraction even at high rank – exactly the
    #    pathology in unslothai/unsloth#4907.  HF's official
    #    ``carla_vlm_gemma.py`` recipe targets these too.
    _DEFAULT_LORA_TARGET_MODULES = (
        r".*\blanguage_model\..*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|gate_up_proj)$"
    )

    def _build_hf_model(self, load_kwargs: dict) -> nn.Module:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)

    def build_messages(self, pil_image, prompt: str, response: str | None = None) -> list[dict]:
        from vidlu_irap_gaim.vlm.models.gemma_utils import build_gemma_chat_messages
        return build_gemma_chat_messages(pil_image, prompt, response=response)

    def tokenize_batch(
        self,
        images: torch.Tensor,
        prompts: list[str],
        responses: list[str],
        targets: torch.Tensor,
        max_length: int = 8192,
    ) -> dict[str, torch.Tensor]:
        """Batched tokenize via a single processor call.

        Strategy: render chat text per sample with ``apply_chat_template(...,
        tokenize=False)`` using **placeholder-only** messages (no PIL image
        embedded), then a single ``processor(text=[...], images=[...],
        padding=True, ...)`` call.  The processor pads ``input_ids``,
        ``attention_mask``, ``mm_token_type_ids``, ``image_position_ids``,
        etc. consistently – manually padding only ``input_ids`` and
        leaving the sequence-aligned extras at per-sample lengths produced
        ``RuntimeError: Sizes of tensors must match except in dimension 0``
        on batches with different sequence lengths.
        """
        batch_size = images.shape[0]
        pil_images = [_to_pil_image(images[i]) for i in range(batch_size)]

        # Placeholder-only messages: the actual PIL is passed via the
        # processor's ``images=`` kwarg below.  Embedding the PIL in the
        # message would conflict with ``images=`` ("got multiple values").
        def _placeholder_msgs(prompt: str, response: str | None = None) -> list[dict]:
            content = [{"type": "image"}, {"type": "text", "text": prompt}]
            msgs = [{"role": "user", "content": content}]
            if response is not None:
                msgs.append({"role": "assistant", "content": response})
            return msgs

        messages_batch = [_placeholder_msgs(prompts[i], responses[i]) for i in range(batch_size)]
        texts_full, texts_prompt_only = self._render_chat_texts(
            messages_batch, _placeholder_msgs(prompts[0])
        )

        # Gemma 4 processor expects images as a list-of-lists (one inner list
        # per text), not a flat list.  Flat ``[pil0, ..., pilN]`` raises
        # ``Received inconsistently sized batches of images (1) and text (N)``
        # because the processor interprets a flat list as a single conversation
        # carrying N images.  Wrap each PIL so the per-text image count is 1.
        inputs = self._processor(
            text=texts_full,
            images=[[pil] for pil in pil_images],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=max_length,
        )
        return self._assemble_tokenized_batch(
            inputs, texts_full, texts_prompt_only, targets, max_length
        )

    def _build_generation_inputs(self, pil_images: list, prompts: list[str]) -> dict:
        return build_gemma_generation_inputs(self._processor, pil_images, prompts,
                                             self._chat_template_kwargs())


# ----- Qwen3.5 ----------------------------------------------------------------

class Qwen35Classifier(Qwen3VLClassifier):
    """Qwen3.5-9B (natively multimodal) with LoRA + QLoRA.

    Despite the different auto class, Qwen3.5 shares Qwen3-VL's *processor*
    contract – ``AutoProcessor`` resolves both to ``Qwen3VLProcessor`` backed by
    ``Qwen2VLImageProcessor``, and ``Qwen3_5ForConditionalGeneration`` is
    registered in transformers' image-text-to-text mapping.  So this subclasses
    ``Qwen3VLClassifier`` and inherits its tokenize/generate path unchanged
    (``qwen_vl_utils.process_vision_info`` included); only the load class,
    attention preference, LoRA targets and thinking mode differ.

    Architecture (``Qwen/Qwen3.5-9B``, config.json):
    - 32 text layers as 8 × (3 × Gated-DeltaNet → 1 × Gated-Attention), i.e.
      ``full_attention_interval=4``, each followed by a **dense** FFN.  This
      checkpoint has no MoE – that is the separate ``qwen3_5_moe`` architecture
      (``Qwen/Qwen3.5-35B-A3B``) – so the Gemma-4 "bitsandbytes silently skips
      MoE experts" pathology does not apply and 4-bit is left on.
    - ``head_dim=256`` on the Gated-Attention blocks.  As with Gemma 4,
      FlashAttention-2 loads but errors at forward on many flash-attn builds at
      head_dim 256, so prefer SDPA → eager.
    - Thinking mode is ON by default in this checkpoint's chat template; see
      ``_DEFAULT_ENABLE_THINKING`` below.

    Requires transformers ≥ 5.x (``transformers.models.qwen3_5``).  The
    Gated-DeltaNet layers additionally use ``causal-conv1d`` and
    ``flash-linear-attention`` when installed; without them transformers falls
    back to a correct but slower and more memory-hungry pure-torch path.
    """

    _DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
    # head_dim=256 on the Gated-Attention blocks → skip flash-attn (see Gemma 4).
    _ATTN_IMPL_PREFS = ("sdpa", "eager")
    # Left on: this checkpoint is dense, so there are no MoE expert linears for
    # bitsandbytes to skip (cf. Gemma4VLClassifier, unslothai/unsloth#4907).
    _DEFAULT_LOAD_IN_4BIT = True
    # Qwen3.5 reasons by default.  Left on, the generation prompt would end on
    # an open `<think>\\n`, the reasoning would compete with the response for the
    # token budget, and the supervised target would carry `</think>`
    # scaffolding (training responses contain no reasoning).  Pass
    # ``enable_thinking=True`` explicitly to opt back in.
    _DEFAULT_ENABLE_THINKING = False
    # The Gated-DeltaNet layers zero the padded positions and then run a causal
    # conv + recurrence over the sequence, which is only correct with the padding
    # on the *right* – left padding would decay the recurrent state before the
    # real tokens start.  This is the same constraint `tokenize_batch` pins
    # `padding_side="right"` for.  Generation needs left padding, so a ragged
    # batch cannot be batched here and falls back to row-by-row; an equal-length
    # batch adds no padding and is unaffected.
    _SUPPORTS_LEFT_PADDED_GENERATION = False
    # The standard attention projections alone would reach only the 8
    # Gated-Attention layers (1 in 4).  The Gated-DeltaNet layers project
    # through `linear_attn.in_proj_{qkv,z,b,a}` / `out_proj` – note PEFT
    # suffix-matches on `.o_proj`, which does NOT match `out_proj` – and every
    # layer has a dense `mlp.{gate,up,down}_proj`.  Verify against a real
    # checkpoint with `vidlu_irap_gaim.tools.dump_module_names`; the base class
    # raises if a target list matches nothing.
    _DEFAULT_LORA_TARGET_MODULES = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def _build_hf_model(self, load_kwargs: dict) -> nn.Module:
        # AutoModelForMultimodalLM per the model card.  It resolves to the same
        # Qwen3_5ForConditionalGeneration as AutoModelForImageTextToText would,
        # since the multimodal-LM mapping is a superset of the latter.
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)
