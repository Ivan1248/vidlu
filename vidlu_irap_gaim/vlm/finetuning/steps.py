"""
Training and evaluation steps for VLM fine-tuning.

Backend-agnostic: the model wrapper (Qwen3VLClassifier / Gemma4VLClassifier /
…) is responsible for the chat template + tokenization (``tokenize_batch``)
and generation + decoding (``generate_for_eval``).  These steps only handle
gradient accumulation, AMP, and parsing the generated text into metric-shaped
predictions.
"""
import dataclasses as dc
import os
import time
import typing as T
import warnings

import torch

from vidlu.utils.collections import NameDict
from vidlu_irap_gaim.vlm.models.generation import bf16_autocast, warn_if_truncated
from vidlu_irap_gaim.vlm.predictions import (
    attribute_predictions_to_one_hot_outputs,
)
from vidlu_irap_gaim.vlm.response_scheme import ResponseScheme

from .dataset import vlm_config_from_data


class _ParsedSample(T.NamedTuple):
    """What `_generate_and_parse_batch` records per sample."""

    predictions: dict
    response: str | None
    thinking: str | None
    is_truncated: bool | None
    error: str | None


def _generate_and_parse_batch(
    model,
    images: torch.Tensor,
    prompts: list[str],
    targets: torch.Tensor,
    response_scheme: ResponseScheme,
    attrs_to_include: list[str],
    max_response_tokens: int = 512,
    amp: bool = True,
) -> NameDict:
    """Generation + parsing for the batch, shaped into metric-ready outputs.

    Delegates the model-specific generate path to ``model.generate_batch_for_eval``,
    which responds to the whole batch in one call rather than one row at a time.

    The generated text is returned alongside the metric-shaped output as
    ``responses``, so it can be inspected without wrapping ``generate_for_eval``.
    ``thinking_texts`` holds the reasoning block that was separated out (``None``
    when thinking is disabled or absent), ``is_truncated_per_row`` is True when the *response*
    used its full token budget without hitting EOS (``None`` when generation
    raised, since no budget was consumed to observe), and ``response_errors``
    the exception string per sample (``None`` when the sample succeeded).
    Because a missing prediction is silently scored as class 0, ``out`` alone
    cannot distinguish a real class-0 prediction from a failure; these fields
    can, and ``is_truncated_per_row`` further distinguishes a format-non-compliant response
    from other failure modes without eyeballing character counts.

    Note ``is_truncated_per_row`` concerns the response only.  Exhausting the *reasoning*
    budget is handled inside the model, which forces the thinking block closed
    and still gives the response its full budget, so it does not surface here.

    A sample whose *generation* raised has ``responses[i] is None``, whereas one
    whose generation succeeded but whose *parsing* raised keeps its text – which
    is what separates a truncated response from a parser bug.  Generation is one
    call for the whole batch, so a failure there is not attributable to a single
    sample and every row gets the same ``response_errors`` entry; parsing stays
    per sample, so a parser failure still names the row it happened on.

    Parsing sees the response with the reasoning already removed, so a response
    scheme never has to cope with a ``</think>`` prefix.
    """
    device = next(model.parameters()).device
    batch_size = images.shape[0]

    generated: list[tuple[str, str | None, bool] | None] = [None] * batch_size
    generation_error = None
    try:
        generated = list(model.generate_batch_for_eval(
            images=[images[i] for i in range(batch_size)],
            prompts=list(prompts),
            max_response_tokens=max_response_tokens,
            amp=amp,
        ))
    except Exception as e:
        # The whole batch shares one generate call, so a failure there is not
        # attributable to a sample; every sample records the same error.
        warnings.warn(f"Generation failed for the batch: {e}")
        generation_error = f"{type(e).__name__}: {e}"

    parsed: list[_ParsedSample] = []
    for i, item in enumerate(generated):
        if item is None:
            parsed.append(_ParsedSample({}, None, None, None, generation_error))
            continue
        response_text, thinking_text, is_truncated = item
        error = None
        try:
            predictions = response_scheme.parse_response(response_text, attrs_to_include)
        except Exception as e:
            warnings.warn(f"Parsing failed for sample {i}: {e}")
            predictions = {}
            error = f"{type(e).__name__}: {e}"
        parsed.append(_ParsedSample(predictions, response_text, thinking_text, is_truncated,
                                    error))
    warn_if_truncated([s.is_truncated for s in parsed], max_response_tokens)

    attr_to_value_to_class_idx = response_scheme.attr_to_value_to_class_idx
    all_attr_names = list(attr_to_value_to_class_idx.keys())
    # Every unusable response is scored as class 0 here, the same as the standalone evaluation's
    # "class0" scoring; the mask that the conversion also returns is only needed by the other.
    out, _ = attribute_predictions_to_one_hot_outputs(
        [s.predictions for s in parsed],
        attr_to_value_to_class_idx,
        all_attr_names,
        device=device,
        attrs_to_include=attrs_to_include,
    )
    return NameDict(out=out, target=targets.to(device),
                    responses=[s.response for s in parsed],
                    thinking_texts=[s.thinking for s in parsed],
                    is_truncated_per_row=[s.is_truncated for s in parsed],
                    response_errors=[s.error for s in parsed])


def _move_tokenized_to_device(tokenized: dict, device) -> dict:
    """Moves tokenizer outputs to ``device`` (keys other than `target` only)."""
    moved = {}
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor) and k != "target":
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


@dc.dataclass
class VLMTrainStep:
    """Training step for VLM fine-tuning with gradient accumulation.

    Backend-agnostic: delegates tokenization to ``trainer.model.tokenize_batch``.
    """

    amp: bool = True
    gradient_accumulation_steps: int = 4

    def __post_init__(self):
        self._accum_count = 0

    def __call__(self, trainer, batch) -> NameDict:
        model = trainer.model
        model.train()
        device = next(model.parameters()).device

        profile_step = os.environ.get("VLM_PROFILE_STEP") == "1"
        t0 = time.perf_counter() if profile_step else None
        tokenized = model.tokenize_batch(
            images=batch["image"],
            prompts=batch["prompt"],
            responses=batch["response"],
            targets=batch["target"],
        )
        t_tok_end = time.perf_counter() if profile_step else None
        t_tok = (t_tok_end - t0) if profile_step else None

        tokenized = _move_tokenized_to_device(tokenized, device)

        autocast = bf16_autocast(self.amp)
        # Forward kwargs: pass everything except `target` (kept for metrics).
        forward_kwargs = {k: v for k, v in tokenized.items() if k != "target"}
        with autocast:
            outputs = model(**forward_kwargs)
            loss = outputs.loss / self.gradient_accumulation_steps

        loss.backward()
        t_gpu_end = time.perf_counter() if profile_step else None
        self._accum_count += 1

        if self._accum_count >= self.gradient_accumulation_steps:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self._accum_count = 0

        if profile_step and t_tok is not None and t_gpu_end is not None and t_tok_end is not None:
            iteration = getattr(trainer.training.state, "iteration", -1)
            if (iteration + 1) % 10 == 0:
                t_gpu = t_gpu_end - t_tok_end
                print(f"tokenize={t_tok:.2f}s gpu={t_gpu:.2f}s")

        return NameDict(
            loss=loss.item() * self.gradient_accumulation_steps,
            out=None,
            target=tokenized["target"],
        )

    def state_dict(self) -> dict:
        return {"_accum_count": self._accum_count}

    def load_state_dict(self, state_dict: dict):
        self._accum_count = state_dict.get("_accum_count", 0)


@dc.dataclass
class VLMEvalStep:
    """Evaluation: teacher-forced loss + generative prediction for metrics.

    Backend-agnostic: delegates both tokenization and generation to the model.

    The step result carries ``responses`` / ``thinking_texts`` / ``is_truncated_per_row`` /
    ``response_errors`` so the decoded text can be inspected without wrapping
    ``generate_for_eval``.  ``EpochLoop`` deletes ``state.result`` right after
    the ``iter_completed`` handlers return, so it has to be read from within an
    evaluation iteration, e.g. ``any(state.result.is_truncated_per_row)`` to check whether
    any response in the batch overran its response budget.
    """

    amp: bool = True
    # Response-token budget.  None ⇒ derive it from the response scheme, which
    # knows the format and the (closed) value set per attribute and can therefore
    # compute a bound no compliant response can exceed – see
    # `ResponseScheme.compute_max_response_tokens`.  A hardcoded number here is a guess
    # about something derivable: 256 was observed cutting a 41-attribute
    # `standard` response off mid-word at line 37, losing every tail attribute to
    # the class-0 fallback in attribute_predictions_to_one_hot_outputs.
    # Set an int only to override the derived bound deliberately.
    max_response_tokens: int | None = None

    def __post_init__(self):
        self._dataset_config = None
        self._response_budget: int | None = None

    @property
    def _response_scheme(self) -> "ResponseScheme":
        return self._dataset_config.response_scheme

    @property
    def _attrs_to_include(self) -> list[str]:
        return self._dataset_config.attrs_to_include

    def get_response_budget(self, model) -> int:
        """Response-token budget for one generated response.

        Reasoning is *not* included: the model budgets it separately and
        guarantees the response its own allowance (see
        ``_BaseVLMClassifier._generate_within_budget``), so reasoning can no longer eat
        the response's tokens and this stays a pure response bound.

        Requires the response scheme, so call only after ``__call__`` has
        resolved it.
        """
        if self.max_response_tokens is not None:
            return self.max_response_tokens
        if self._response_budget is None:
            self._response_budget = self._response_scheme.compute_max_response_tokens(
                model.tokenizer, self._attrs_to_include
            )
        return self._response_budget

    def _ensure_dataset_config(self, trainer):
        if self._dataset_config is None:
            self._dataset_config = vlm_config_from_data(trainer.data)

    def __call__(self, trainer, batch) -> NameDict:
        self._ensure_dataset_config(trainer)

        model = trainer.model
        model.eval()
        device = next(model.parameters()).device

        # 1. Teacher-forced loss
        tokenized = model.tokenize_batch(
            images=batch["image"],
            prompts=batch["prompt"],
            responses=batch["response"],
            targets=batch["target"],
        )
        tokenized = _move_tokenized_to_device(tokenized, device)

        autocast = bf16_autocast(self.amp)
        forward_kwargs = {k: v for k, v in tokenized.items() if k != "target"}
        with torch.no_grad(), autocast:
            outputs = model(**forward_kwargs)
            loss = outputs.loss.item()

        # 2. Generative prediction for real metrics.  Skipped when
        # VLM_SKIP_GENERATIVE_EVAL=1 – defer to scripts/eval_vlm_checkpoint.py
        # for a one-off post-training pass (per-sample autoregressive decode
        # otherwise dominates eval wall-time).
        if os.environ.get("VLM_SKIP_GENERATIVE_EVAL") == "1":
            return NameDict(
                out=None,
                target=tokenized["target"],
                loss=loss,
                responses=None,
                thinking_texts=None,
                is_truncated_per_row=None,
                response_errors=None,
            )
        gen_result = _generate_and_parse_batch(
            model=model,
            images=batch["image"],
            prompts=batch["prompt"],
            targets=batch["target"],
            response_scheme=self._response_scheme,
            attrs_to_include=self._attrs_to_include,
            max_response_tokens=self.get_response_budget(model),
            amp=self.amp,
        )
        return NameDict(out=gen_result.out, target=gen_result.target, loss=loss,
                        responses=gen_result.responses,
                        thinking_texts=gen_result.thinking_texts,
                        is_truncated_per_row=gen_result.is_truncated_per_row,
                        response_errors=gen_result.response_errors)
