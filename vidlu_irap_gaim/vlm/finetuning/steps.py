"""
Training and evaluation steps for VLM fine-tuning.

These steps integrate with Vidlu's Trainer class and handle:
- Tokenization (using processor from model)
- Gradient accumulation for memory efficiency
- AMP (automatic mixed precision) support
- Generative evaluation with response parsing for real F1/accuracy metrics
"""
from contextlib import nullcontext
import dataclasses as dc
import os
import time
import warnings

import torch
from transformers import AutoProcessor

from vidlu.utils.collections import NameDict
from vidlu_irap_gaim.vlm.base import _to_pil_image
from vidlu_irap_gaim.vlm.response_scheme import ResponseScheme
from vidlu_irap_gaim.vlm.response_parser import convert_attribute_predictions_to_standard_format
from vidlu_irap_gaim.vlm.qwen_utils import build_qwen_chat_messages



def _create_labels_with_prompt_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_positions: list[int],
) -> torch.Tensor:
    """Create labels tensor with prompt and padding tokens masked to -100.

    Args:
        input_ids: (batch_size, seq_len) token IDs.
        attention_mask: (batch_size, seq_len) attention mask.
        response_start_positions: Index in the (possibly padded) sequence where
            the response starts for each sample. Accounts for leading padding so
            the result is correct for both left-padded and right-padded batches.

    Returns:
        Labels tensor with prompt tokens set to -100.
    """
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
    """Compute the index in each padded sequence where response tokens begin.

    Avoids a second full processor() call by tokenizing only the short
    response-in-template suffix per sample (the part of texts_full[i] that
    follows texts_prompt_only[i]).  Handles both left-padded and right-padded
    batches correctly.

    Args:
        attention_mask: (batch_size, seq_len) from the full tokenized batch.
        texts_full: Template-applied texts including the response (one per sample).
        texts_prompt_only: Template-applied texts without the response, ending
            with the generation prompt marker (one per sample).
            Must satisfy: texts_prompt_only[i] is a prefix of texts_full[i].
        tokenizer: HuggingFace tokenizer from the processor.

    Returns:
        List of response-start indices in the (padded) input_ids sequences.
    """
    positions = []
    for mask, full_text, prompt_only_text in zip(attention_mask, texts_full, texts_prompt_only):
        # The response suffix is the portion of the full template text that follows
        # the prompt-only prefix (i.e., the assistant response + end-of-turn markers).
        response_in_template_text = full_text[len(prompt_only_text):]
        response_in_template_len = len(
            tokenizer.encode(response_in_template_text, add_special_tokens=False)
        )

        # Prompt-only token count (accounts for variable image token counts per sample)
        full_nonpadded_len = int(mask.sum())
        prompt_only_len = full_nonpadded_len - response_in_template_len

        # argmax on a binary mask returns the first True position:
        # 0 for right-padded batches, pad_count for left-padded batches.
        first_real_token = int((mask == 1).long().argmax())
        positions.append(first_real_token + prompt_only_len)
    return positions


def _tokenize_batch(
    images: torch.Tensor,
    prompts: list[str],
    responses: list[str],
    targets: torch.Tensor,
    processor: "AutoProcessor",
    max_length: int = 8192,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch for VLM training.

    Args:
        images: (B, C, H, W) batch of image tensors.
        prompts: List of prompt strings.
        responses: List of response strings (ground truth).
        targets: (B, A) batch of target tensors for metrics.
        processor: HuggingFace processor with tokenizer and image processor.
        max_length: Maximum sequence length.  Must be large enough to fit
            image tokens + prompt + response.  Qwen3-VL supports up to 32k.

    Returns:
        Dictionary with tokenized inputs ready for the model:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - pixel_values: (batch_size, ...) image features
        - labels: (batch_size, seq_len) with prompt tokens masked
        - target: (batch_size, num_attrs) original targets for metrics
    """
    from qwen_vl_utils import process_vision_info

    batch_size = images.shape[0]

    # Build chat messages for each sample (with response appended for training).
    messages_batch = [
        build_qwen_chat_messages(_to_pil_image(images[i]), prompts[i], response=responses[i])
        for i in range(batch_size)
    ]

    # Get image inputs (extracted once from messages_batch)
    all_image_inputs = []
    for m in messages_batch:
        img_inputs, _ = process_vision_info(m)
        all_image_inputs.extend(img_inputs if img_inputs else [])

    # Build full text (prompt + image + response) for training
    texts_full = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages_batch
    ]

    # Build prompt-only text to locate the response boundary.
    # Drop the assistant message (last element). VLMBihDataset uses identical prompt
    # for all samples; the template output differs only by image placeholder, which
    # is the same string. Build once and reuse.
    text_prompt_only = processor.apply_chat_template(
        messages_batch[0][:-1], tokenize=False, add_generation_prompt=True
    )
    texts_prompt_only = [text_prompt_only] * batch_size
    if os.environ.get("VLM_VERIFY_PROMPT_ONLY") == "1":
        per_sample = [
            processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        assert len(set(per_sample)) == 1 and per_sample[0] == text_prompt_only, (
            "prompt-only template output differs across samples"
        )

    # Single processor call: tokenize full sequences and process images once
    inputs = processor(
        text=texts_full,
        images=all_image_inputs if all_image_inputs else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Compute response start positions without a second processor call.
    # Only the short response-in-template suffix is tokenized per sample.
    response_start_positions = _compute_response_start_positions(
        attention_mask=inputs["attention_mask"],
        texts_full=texts_full,
        texts_prompt_only=texts_prompt_only,
        tokenizer=processor.tokenizer,
    )

    # Create labels with prompt masking
    labels = _create_labels_with_prompt_mask(
        inputs["input_ids"],
        inputs["attention_mask"],
        response_start_positions,
    )

    # Validate that some response tokens survived truncation
    num_valid_labels = (labels != -100).sum().item()
    if num_valid_labels == 0:
        seq_len = inputs["input_ids"].shape[1]
        raise RuntimeError(
            f"All labels are masked (-100): the response was completely truncated. "
            f"seq_len={seq_len}, max_length={max_length}, "
            f"response_start_positions={response_start_positions}. "
            f"Increase max_length or reduce prompt length "
            f"(fewer attributes / lower detail_level)."
        )

    result = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        "target": targets,
    }

    # Add pixel_values if present
    if "pixel_values" in inputs:
        result["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        result["image_grid_thw"] = inputs["image_grid_thw"]

    return result


def _extract_response_scheme_from_data(data: dict) -> ResponseScheme:
    """Extract the ResponseScheme stored in dataset info.

    Looks for ``vlm_response_scheme`` on the info of the first split found in
    ``data``.  Raises a clear error if the dataset was not created via
    ``make_vlm_bih_data`` (which is the only supported source).

    Args:
        data: Mapping of split name -> dataset (as held by Trainer.data).

    Returns:
        ResponseScheme stored in the dataset info.

    Raises:
        RuntimeError: If no dataset with vlm_response_scheme info is found.
    """
    for split_name, dataset in data.items():
        info = getattr(dataset, "info", None)
        scheme = getattr(info, "vlm_response_scheme", None)
        if scheme is not None:
            return scheme
    available = list(data.keys())
    raise RuntimeError(
        f"VLMEvalStep could not find vlm_response_scheme in any dataset split {available}. "
        "Ensure the dataset was created with make_vlm_bih_data()."
    )


def _generate_and_parse_batch(
    model,
    images: torch.Tensor,
    prompts: list[str],
    targets: torch.Tensor,
    response_scheme: ResponseScheme,
    attrs_to_include: list[str],
    max_new_tokens: int = 512,
    amp: bool = True,
) -> NameDict:
    """Generate responses for a batch and parse into metric-compatible predictions.

    Performs per-sample autoregressive generation, parses the VLM text responses
    via ``response_scheme``, and builds one-hot output tensors compatible with
    MultiAttributeClassificationMetrics.

    Args:
        model: Qwen3VLClassifier with loaded model and processor.
        images: (B, C, H, W) batch of image tensors.
        prompts: List of prompt strings (one per sample).
        targets: (B, A) target tensor with class indices per attribute.
        response_scheme: ResponseScheme instance used for parsing.
        attrs_to_include: Attribute names to classify.
        max_new_tokens: Maximum tokens to generate per sample.
        amp: Whether to use automatic mixed precision.

    Returns:
        NameDict with:
        - out: list of (B, K_i) tensors per attribute (one-hot predictions)
        - target: (B, A) target tensor on device
    """
    from qwen_vl_utils import process_vision_info

    processor = model.processor
    device = next(model.parameters()).device
    batch_size = images.shape[0]

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()

    # Per-sample generation and parsing
    batch_predictions = []
    for i in range(batch_size):
        pil_image = _to_pil_image(images[i])
        messages = build_qwen_chat_messages(pil_image, prompts[i])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        try:
            with torch.no_grad(), amp_ctx:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            input_len = inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_len:]
            response_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            predictions = response_scheme.parse_response(response_text, attrs_to_include)
        except Exception as e:
            warnings.warn(f"Generation/parsing failed for sample {i}: {e}")
            predictions = {}

        batch_predictions.append(predictions)


    attr_to_value_to_class_idx = response_scheme.attr_to_value_to_class_idx
    all_attr_names = list(attr_to_value_to_class_idx.keys())
    out_list = convert_attribute_predictions_to_standard_format(
        batch_predictions,
        attr_to_value_to_class_idx,
        all_attr_names,
        device=device,
        attrs_to_include=attrs_to_include,
    )

    return NameDict(
        out=out_list,
        target=targets.to(device),
    )


@dc.dataclass
class VLMTrainStep:
    """Training step for VLM fine-tuning with gradient accumulation.

    This step handles:
    - Tokenization using processor from model
    - Forward pass with VLM model
    - Loss computation (cross-entropy on response tokens)
    - Gradient accumulation for memory efficiency
    - AMP support for mixed precision training

    The step consumes pre-built ``batch["prompt"]`` and ``batch["response"]``
    strings produced by VLMBihDataset. It does not need to know the
    ResponseScheme directly — the format is already baked into those strings.

    Args:
        amp: Whether to use automatic mixed precision.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
    """

    amp: bool = True
    gradient_accumulation_steps: int = 4

    def __post_init__(self):
        self._accum_count = 0

    def __call__(self, trainer, batch) -> NameDict:
        """Execute one training step.

        Args:
            trainer: Vidlu Trainer instance.
            batch: Collated batch with image, prompt, response, target fields.

        Returns:
            NameDict with loss, out (metric-compatible dummy outputs), and target.
        """
        model = trainer.model
        model.train()

        device = next(model.parameters()).device

        # Tokenize batch using processor from model
        profile_step = os.environ.get("VLM_PROFILE_STEP") == "1"
        t0 = time.perf_counter() if profile_step else None
        tokenized = _tokenize_batch(
            images=batch["image"],  # (B, C, H, W)
            prompts=batch["prompt"],  # list[str]
            responses=batch["response"],  # list[str]
            targets=batch["target"],  # (B, A)
            processor=model.processor,
        )
        t_tok_end = time.perf_counter() if profile_step else None
        t_tok = (t_tok_end - t0) if profile_step else None

        # Move to device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        labels = tokenized["labels"].to(device)

        pixel_values = tokenized.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        image_grid_thw = tokenized.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        # Forward pass with AMP
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

        with amp_ctx:
            kwargs = {}
            if image_grid_thw is not None:
                kwargs["image_grid_thw"] = image_grid_thw

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )
            loss = outputs.loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()
        t_gpu_end = time.perf_counter() if profile_step else None
        self._accum_count += 1

        # Optimizer step after accumulation
        if self._accum_count >= self.gradient_accumulation_steps:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self._accum_count = 0

        if profile_step and t_tok is not None and t_gpu_end is not None and t_tok_end is not None:
            # iteration is from previous step; current batch index = iteration + 1
            iteration = getattr(trainer.training.state, "iteration", -1)
            if (iteration + 1) % 10 == 0:
                t_gpu = t_gpu_end - t_tok_end
                print(f"tokenize={t_tok:.2f}s gpu={t_gpu:.2f}s")

        return NameDict(
            loss=loss.item() * self.gradient_accumulation_steps,
            out=None,  # actual prediction not available due to teacher forcing
            target=tokenized["target"],
        )

    def state_dict(self) -> dict:
        """Save step state for checkpointing."""
        return {"_accum_count": self._accum_count}

    def load_state_dict(self, state_dict: dict):
        """Load step state from checkpoint."""
        self._accum_count = state_dict.get("_accum_count", 0)


@dc.dataclass
class VLMEvalStep:
    """Evaluation step with generative prediction for real metric computation.

    For each batch, this step:
    1. Computes cross-entropy loss (fast teacher-forced forward pass)
    2. Generates text responses and parses them into attribute predictions

    The parsed predictions produce real F1/accuracy metrics, unlike the
    training step's dummy outputs.

    Note: Generative evaluation is slower than loss-only due to autoregressive
    decoding (~2-5 s/sample). Control evaluation frequency and set size via
    eval_count and eval_batch_size in TrainerConfig.

    Args:
        amp: Whether to use automatic mixed precision.
        max_new_tokens: Maximum tokens to generate per sample.
    """

    amp: bool = True
    max_new_tokens: int = 512

    def __post_init__(self):
        self._response_scheme: "ResponseScheme | None" = None
        self._attrs_to_include: list[str] | None = None

    def _ensure_response_scheme(self, trainer):
        """Derive the ResponseScheme from the trainer's dataset info."""
        if self._response_scheme is not None:
            return
        self._response_scheme = _extract_response_scheme_from_data(trainer.data)

    def _ensure_attrs_to_include(self):
        """Lazily load the attribute list."""
        if self._attrs_to_include is not None:
            return
        from vidlu_irap_gaim.attrs import get_attrs_to_include

        self._attrs_to_include = list(get_attrs_to_include())

    def __call__(self, trainer, batch) -> NameDict:
        """Execute evaluation with loss computation and generative prediction.

        Args:
            trainer: Vidlu Trainer instance.
            batch: Collated batch with image, prompt, response, target fields.

        Returns:
            NameDict with loss, out (real parsed predictions), and target.
        """
        self._ensure_response_scheme(trainer)
        self._ensure_attrs_to_include()

        model = trainer.model
        model.eval()

        device = next(model.parameters()).device

        # 1. Compute loss via teacher-forced forward pass (fast)
        tokenized = _tokenize_batch(
            images=batch["image"],
            prompts=batch["prompt"],
            responses=batch["response"],
            targets=batch["target"],
            processor=model.processor,
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        labels = tokenized["labels"].to(device)

        pixel_values = tokenized.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        image_grid_thw = tokenized.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

        with torch.no_grad(), amp_ctx:
            kwargs = {}
            if image_grid_thw is not None:
                kwargs["image_grid_thw"] = image_grid_thw
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )
            loss = outputs.loss.item()

        # 2. Generate and parse for real metric computation (slower)
        gen_result = _generate_and_parse_batch(
            model=model,
            images=batch["image"],
            prompts=batch["prompt"],
            targets=batch["target"],
            response_scheme=self._response_scheme,
            attrs_to_include=self._attrs_to_include,
            max_new_tokens=self.max_new_tokens,
            amp=self.amp,
        )

        return NameDict(
            out=gen_result.out,
            target=gen_result.target,
            loss=loss,
        )
