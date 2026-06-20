"""
Training and evaluation steps for VLM fine-tuning.

Backend-agnostic: the model wrapper (Qwen3VLClassifier / Gemma4VLClassifier /
…) is responsible for the chat template + tokenization (``tokenize_batch``)
and generation + decoding (``generate_for_eval``).  These steps only handle
gradient accumulation, AMP, and parsing the generated text into metric-shaped
predictions.
"""
from contextlib import nullcontext
import dataclasses as dc
import os
import time
import warnings

import torch

from vidlu.utils.collections import NameDict
from vidlu_irap_gaim.vlm.response_scheme import ResponseScheme
from vidlu_irap_gaim.vlm.predictions import convert_attribute_predictions_to_standard_format


def _extract_response_scheme_from_data(data: dict) -> ResponseScheme:
    """Find the ResponseScheme on dataset.info (set by make_vlm_bih_data)."""
    for split_name, dataset in data.items():
        info = getattr(dataset, "info", None)
        scheme = getattr(info, "vlm_response_scheme", None)
        if scheme is not None:
            return scheme
    raise RuntimeError(
        f"VLMEvalStep could not find vlm_response_scheme in any dataset split "
        f"{list(data.keys())}. Ensure the dataset was created with make_vlm_bih_data()."
    )


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
    """Per-sample generation + parsing, batched into metric-shaped outputs.

    Delegates the model-specific generate path to ``model.generate_for_eval``.
    """
    device = next(model.parameters()).device
    batch_size = images.shape[0]

    batch_predictions = []
    for i in range(batch_size):
        try:
            response_text = model.generate_for_eval(
                image=images[i],
                prompt=prompts[i],
                max_response_tokens=max_response_tokens,
                amp=amp,
            )
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
    return NameDict(out=out_list, target=targets.to(device))


def _move_tokenized_to_device(tokenized: dict, device) -> dict:
    """Move tokenizer outputs to ``device`` (keys other than `target` only)."""
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

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        # Forward kwargs: pass everything except `target` (kept for metrics).
        forward_kwargs = {k: v for k, v in tokenized.items() if k != "target"}
        with amp_ctx:
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
    """

    amp: bool = True
    # 256 is sufficient for BIH attribute schemes; 512 was paying for a
    # generation cap that responses don't approach.
    max_response_tokens: int = 256

    def __post_init__(self):
        self._response_scheme: "ResponseScheme | None" = None
        self._attrs_to_include: list[str] | None = None

    def _ensure_response_scheme(self, trainer):
        if self._response_scheme is None:
            self._response_scheme = _extract_response_scheme_from_data(trainer.data)

    def _ensure_attrs_to_include(self):
        if self._attrs_to_include is None:
            from irap_data.attrs import get_attrs_to_include
            self._attrs_to_include = list(get_attrs_to_include())

    def __call__(self, trainer, batch) -> NameDict:
        self._ensure_response_scheme(trainer)
        self._ensure_attrs_to_include()

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

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
        forward_kwargs = {k: v for k, v in tokenized.items() if k != "target"}
        with torch.no_grad(), amp_ctx:
            outputs = model(**forward_kwargs)
            loss = outputs.loss.item()

        # 2. Generative prediction for real metrics.  Skipped when
        # VLM_SKIP_GENERATIVE_EVAL=1 — defer to scripts/eval_generative_gemma.py
        # for a one-off post-training pass (per-sample autoregressive decode
        # otherwise dominates eval wall-time).
        if os.environ.get("VLM_SKIP_GENERATIVE_EVAL") == "1":
            return NameDict(
                out=None,
                target=tokenized["target"],
                loss=loss,
            )
        gen_result = _generate_and_parse_batch(
            model=model,
            images=batch["image"],
            prompts=batch["prompt"],
            targets=batch["target"],
            response_scheme=self._response_scheme,
            attrs_to_include=self._attrs_to_include,
            max_response_tokens=self.max_response_tokens,
            amp=self.amp,
        )
        return NameDict(out=gen_result.out, target=gen_result.target, loss=loss)
