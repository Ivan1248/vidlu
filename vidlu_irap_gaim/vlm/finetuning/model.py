"""
VLM classifier wrappers (Qwen3-VL, Gemma 4) for Vidlu training integration.

Each classifier:
- Loads its base HF model eagerly in initialize() so parameters exist before
  the optimizer is created.
- Attaches a LoRA adapter (PEFT) so only the adapter weights are trainable.
- Optionally uses 4-bit NF4 quantization for the base weights (QLoRA).
- Saves an adapter-only state_dict (~100s of MB instead of ~10s of GB).
- Exposes ``build_messages``, ``tokenize_batch``, and ``generate_for_eval``
  so ``VLMTrainStep`` / ``VLMEvalStep`` stay backend-agnostic.
"""

from typing import Any
import os

import torch
from torch import nn

from vidlu_irap_gaim.vlm.image_utils import to_pil_image as _to_pil_image


# ----- shared tokenization helpers (used by subclass tokenize_batch) ----------

def _create_labels_with_prompt_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_positions: list[int],
) -> torch.Tensor:
    """Clone input_ids and mask prompt and padding tokens to -100 for CE loss."""
    labels = input_ids.clone()
    for i, response_start in enumerate(response_start_positions):
        labels[i, :response_start] = -100
        labels[i, attention_mask[i] == 0] = -100
    return labels


def _compute_response_start_positions(
    attention_mask: torch.Tensor,
    texts_full: list[str],
    texts_prompt_only: list[str],
    tokenizer,
) -> list[int]:
    """Per-sample index in the padded sequence where assistant response begins.

    Handles both left- and right-padded batches by locating the first
    attended token via ``argmax(attention_mask == 1)``.
    """
    positions = []
    for mask, full_text, prompt_only_text in zip(attention_mask, texts_full, texts_prompt_only):
        response_in_template_text = full_text[len(prompt_only_text):]
        response_in_template_len = len(
            tokenizer.encode(response_in_template_text, add_special_tokens=False)
        )
        full_nonpadded_len = int(mask.sum())
        prompt_only_len = full_nonpadded_len - response_in_template_len
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
      - ``generate_for_eval(image, prompt, max_new_tokens, amp)`` : single-sample
        autoregressive generation that returns the decoded response text.
    """

    _DEFAULT_MODEL_ID: str = ""
    # tuple[str, ...] → PEFT suffix-matches each entry against module names.
    # str            → PEFT treats it as a regex (re.fullmatch on full path).
    _DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] | str = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    # Order in which attention implementations are tried at load time.
    # The first one that loads AND survives the first forward is used.
    # Subclasses override this when the default would fail at forward time
    # (e.g. Gemma 4 has head_dim=256 which trips many flash-attn builds).
    _ATTN_IMPL_PREFS: tuple[str, ...] = ("flash_attention_2", "sdpa", "eager")
    # NF4 quantization of the base weights.  True works for plain (non-MoE)
    # VLMs (Qwen3-VL); some MoE models (Gemma 4) need bf16 instead.
    _DEFAULT_LOAD_IN_4BIT: bool = True
    # HF device_map for from_pretrained.
    #   None   → ``{"": self._device or "cuda"}`` (single-device load).
    #   "auto" → HF places shards across visible GPUs (naive model parallel).
    _DEFAULT_DEVICE_MAP: dict | str | None = None

    def __init__(
        self,
        model_id: str | None = None,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] | str | None = None,
        load_in_4bit: bool | None = None,
        use_gradient_checkpointing: bool = True,
        device_map: dict | str | None = None,
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
        self.load_in_4bit = (
            self._DEFAULT_LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # device_map=None here means "use class default" (which may itself be
        # None ⇒ single-device fallback computed at load time).
        self.device_map = self._DEFAULT_DEVICE_MAP if device_map is None else device_map
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

    def generate_for_eval(
        self,
        image: torch.Tensor,
        prompt: str,
        max_response_tokens: int = 512,
        amp: bool = True,
    ) -> str:
        raise NotImplementedError

    # --- shared tokenize helpers ----------------------------------------------

    def _render_chat_texts(
        self,
        messages_full_batch: list[list[dict]],
        messages_prompt_only_template: list[dict],
    ) -> tuple[list[str], list[str]]:
        """Render full and (broadcast) prompt-only chat texts for the batch.

        VLMIrapDataset uses an identical prompt for every sample, so the
        prompt-only template text is computed once and broadcast across the
        batch (measurably faster than per-sample apply). Under
        ``VLM_VERIFY_PROMPT_ONLY=1`` the broadcast assumption is checked.
        """
        processor = self._processor
        texts_full = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_full_batch
        ]
        text_prompt_only = processor.apply_chat_template(
            messages_prompt_only_template, tokenize=False, add_generation_prompt=True
        )
        texts_prompt_only = [text_prompt_only] * len(messages_full_batch)
        if os.environ.get("VLM_VERIFY_PROMPT_ONLY") == "1":
            per_sample = [
                processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
                for m in messages_full_batch
            ]
            assert len(set(per_sample)) == 1 and per_sample[0] == text_prompt_only, (
                "prompt-only template output differs across samples"
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
        """Compute labels and forward every tensor produced by the processor."""
        response_start_positions = _compute_response_start_positions(
            attention_mask=inputs["attention_mask"],
            texts_full=texts_full,
            texts_prompt_only=texts_prompt_only,
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

    def _load(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, BitsAndBytesConfig

        if self._loaded:
            return

        device = self._device or "cuda"

        if "HF_HUB_DISABLE_XET" not in os.environ:
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        cls_name = type(self).__name__
        print(f"[{cls_name}] Loading {self.model_id}...")

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        device_map = self.device_map if self.device_map is not None else {"": device}
        load_kwargs = dict(
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
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

        # PEFT: str → regex match, list → suffix match.
        target_modules = (
            self.lora_target_modules if isinstance(self.lora_target_modules, str)
            else list(self.lora_target_modules)
        )
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, peft_config)

        # Catch the silent failure mode where the LoRA target names do not
        # match any sublayer (PEFT then produces an adapter with zero trainable
        # params).  Important for new model families like Gemma 4 where the
        # default targets may not apply.
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError(
                f"[{cls_name}] LoRA wrapping produced 0 trainable parameters. "
                f"lora_target_modules={self.lora_target_modules} matched nothing. "
                f"Run `print({{name for name,_ in self._model.named_modules()}})` "
                f"to find the right names for this model."
            )

        if self.use_gradient_checkpointing:
            # use_cache must be False with gradient checkpointing.
            self._model.config.use_cache = False
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # Silences a per-sample "Setting pad_token_id to eos_token_id" warning from
        # generate(). eos is the value HF falls back to, so outputs are unaffected.
        gen_cfg = getattr(self._model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
            gen_cfg.pad_token_id = pad_id

        self._loaded = True

        total_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"[{cls_name}] Loaded. Trainable: {trainable_params:,} / "
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
        """Return only LoRA adapter weights + config metadata.

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
        return {
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
            truncation=True,
            max_length=max_length,
        )
        return self._assemble_tokenized_batch(
            inputs, texts_full, texts_prompt_only, targets, max_length
        )

    def generate_for_eval(
        self,
        image: torch.Tensor,
        prompt: str,
        max_response_tokens: int = 512,
        amp: bool = True,
    ) -> str:
        from contextlib import nullcontext
        from qwen_vl_utils import process_vision_info

        processor = self._processor
        device = next(self._model.parameters()).device

        pil_image = _to_pil_image(image)
        messages = self.build_messages(pil_image, prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            # Translate project's `max_response_tokens` to HF's documented kwarg.
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_response_tokens)
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


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
    # on many flash-attn builds — Gemma 4 text has head_dim=256 which is on
    # the boundary and trips a strict check.  Keep eager as the final fallback.
    _ATTN_IMPL_PREFS = ("sdpa", "eager")
    # bnb 4-bit silently skips this model's MoE expert linears (observed:
    # load consumes ~48 GB on one A6000, i.e. the experts stay bf16).  See
    # unslothai/unsloth#4907.  Default to bf16 and shard across visible GPUs
    # via ``device_map="auto"`` (naive model parallel on 4× A6000 fits
    # comfortably at ~12.5 GB/GPU for the base weights).
    _DEFAULT_LOAD_IN_4BIT = False
    _DEFAULT_DEVICE_MAP = "auto"
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
    #    small trainable-param fraction even at high rank — exactly the
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
        etc. consistently — manually padding only ``input_ids`` and
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
            truncation=True,
            max_length=max_length,
        )
        return self._assemble_tokenized_batch(
            inputs, texts_full, texts_prompt_only, targets, max_length
        )

    def generate_for_eval(
        self,
        image: torch.Tensor,
        prompt: str,
        max_response_tokens: int = 512,
        amp: bool = True,
    ) -> str:
        from contextlib import nullcontext

        processor = self._processor
        device = next(self._model.parameters()).device

        pil_image = _to_pil_image(image)
        messages = self.build_messages(pil_image, prompt)
        # Image is embedded in messages; do NOT pass images= (see gemma_utils).
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            # Translate project's `max_response_tokens` to HF's documented kwarg.
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_response_tokens)
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]
