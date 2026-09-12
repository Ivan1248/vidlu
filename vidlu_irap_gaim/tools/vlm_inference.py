"""
VLM road attribute classification by generation, with metrics.

Drives any `BaseVLMPredictor` over a dataset, one *session* -- an independent
(image, prompt) -> response exchange -- per attribute group. ``attrs_per_session``
is the knob: ``None`` asks for every attribute in one prompt (what fine-tuning
trains against), ``1`` gives every attribute its own prompt and its own session.

Prompts and scoring come from the dataset's own ``info`` (``vlm_response_scheme``,
``vlm_attrs_to_include``, set by ``make_vlm_bih_data``), the same source the
fine-tuning eval step reads, so a zero-shot run and a fine-tuned run cannot end
up prompting or scoring differently.

Usage (CLI):
    # Evaluate on test split with ground truth, all attributes in one session
    IRAP_HOME=/path/to/data python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --output-dir results/vlm_zeroshot

    # One session and one prompt per attribute
    IRAP_HOME=/path/to/data python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --attrs-per-session 1 --output-dir results/vlm_per_attr

    # Evaluate on custom folder (no labels)
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --image-folder /path/to/images --output-dir results/vlm_custom

    # Use FP8 model with vLLM backend (auto-selected for FP8 models)
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --model-id "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

Usage (via Vidlu runner):
    python scripts/run.py test <data> <input_adapter> <model> <trainer> \\
        -m irap_gaim.tools.vlm_inference:run,e,attrs_per_session=1

To evaluate a `_BaseVLMClassifier` (pretrained or fine-tuned) rather than a
standalone predictor, see ``vidlu_irap_gaim.vlm.finetuning.predictor``.

Interactive control:
    Type "skip" during evaluation to stop early and save partial results.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from vidlu_irap_gaim.vlm.models.base import attribute_sessions
from vidlu_irap_gaim.vlm.models.thinking import DEFAULT_THINKING_BUDGET
from vidlu_irap_gaim.vlm.predictions import is_usable_prediction
from vidlu_irap_gaim.vlm.response_scheme import DEFAULT_RESPONSE_TOKEN_MARGIN
from vidlu_irap_gaim.vlm.scoring import (count_scored_and_invalid_responses,
                                         metrics_to_json_dict, print_metrics,
                                         update_both_scorings)

# Prompt tokens a batch aims for. Prefill is compute-bound and saturates at a
# roughly constant *token* count, not a constant number of images, so sizing the
# batch by tokens keeps the GPU equally busy whether a session asks about one
# attribute (a short prompt) or 41 (a long one) -- a fixed image count would
# saturate one and starve the other.
DEFAULT_BATCH_TOKENS = 16384
# Cap on images per batch regardless of token budget, bounding the KV cache.
MAX_BATCH_SIZE = 64


@dataclass
class EvaluationResult:
    """Summary of VLM evaluation results."""

    num_samples_requested: int
    num_samples_completed: int
    num_valid_predictions: int
    metrics: dict[str, float] | None
    metrics_excluding_invalid: dict[str, float] | None
    num_invalid_attribute_responses: int
    num_scored_attribute_responses: int
    num_truncated_sessions: int
    num_sessions: int
    duration_s: float
    output_dir: Path
    predictions_file: Path
    records_file: Path
    was_interrupted: bool = False

    @property
    def invalid_rate(self) -> float:
        """Fraction of (segment, attribute) responses that were unusable."""
        if self.num_scored_attribute_responses == 0:
            return float("nan")
        return self.num_invalid_attribute_responses / self.num_scored_attribute_responses

    @property
    def truncation_rate(self) -> float:
        """Fraction of sessions that hit the response budget without stopping.

        Must be ~0 for a comparison between session granularities to mean
        anything: a budget that binds in one arm only manufactures a difference.
        """
        if self.num_sessions == 0:
            return float("nan")
        return self.num_truncated_sessions / self.num_sessions


def make_eval_data(
    dataset_name: str = "bih",
    response_scheme: str = "standard",
    detail_level: str | None = None,
    upsampling_factor: int = 1,
) -> dict:
    """Builds the VLM-wrapped splits the evaluation reads its prompts from.

    The same factory the fine-tuning run uses, so the standalone CLI and a
    `run.py` experiment evaluate identical datasets with identical prompts.
    """
    from vidlu_irap_gaim.vlm.finetuning.dataset import (
        make_vlm_bih_data, make_vlm_vietnam_data)
    from vidlu_irap_gaim.vlm.prompts import DEFAULT_DETAIL_LEVEL

    make = {"bih": make_vlm_bih_data, "vietnam": make_vlm_vietnam_data}[dataset_name]
    return make(response_scheme=response_scheme,
                detail_level=detail_level or DEFAULT_DETAIL_LEVEL,
                upsampling_factor=upsampling_factor)


def _load_dataset(
    split: str,
    image_folder: str | None,
    dataset_name: str = "bih",
    response_scheme: str = "standard",
    detail_level: str | None = None,
    upsampling_factor: int = 1,
):
    """Loads a dataset for evaluation.

    Returns:
        The dataset. Labeled splits are VLM-wrapped, so they carry the response
        scheme and the attribute subset on ``info``; an unlabeled image folder
        carries only the attribute metadata, and the caller supplies the rest.
    """
    if image_folder is None:
        return make_eval_data(dataset_name, response_scheme, detail_level,
                              upsampling_factor)[split]

    from irap_data import InferenceImageDataset

    # Custom folder - needs a reference dataset for the attribute metadata.
    ref_ds = make_eval_data(dataset_name, response_scheme, detail_level,
                            upsampling_factor)["test"]
    return InferenceImageDataset.from_folder(image_folder, reference_dataset=ref_ds)


def _dataset_has_labels(dataset: Any) -> bool:
    try:
        sample = dataset[0]
    except Exception:
        return False
    return hasattr(sample, "keys") and "target" in sample.keys()


def _extract_image(sample):
    """The image to classify, from a VLM-wrapped or a raw iRAP sample.

    ``VLMIrapDataset`` already picks the center frame and yields it as ``image``;
    a raw sample still carries the whole ``rgb`` sequence.
    """
    if hasattr(sample, "keys"):
        if "image" in sample.keys():
            return sample["image"]
        rgb = sample["rgb"]
    else:
        rgb = sample[0]
    return rgb[0] if rgb.ndim == 4 else rgb


def _get_segment_id(sample, idx):
    return sample.get("segment_id", str(idx)) if hasattr(sample, "get") else str(idx)


def resolve_batch_size(
    predictor,
    prompt: str,
    batch_size: int | None,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
) -> int:
    """Images per generation batch, sized from a token budget when not given.

    See ``DEFAULT_BATCH_TOKENS`` for why the budget is in tokens. The image's
    token count is not known without preprocessing it, so it is approximated by
    a constant; being off by a factor here costs throughput, never correctness.
    """
    if batch_size is not None:
        return max(1, int(batch_size))
    approx_image_tokens = 512
    prompt_tokens = len(predictor.tokenizer.encode(prompt, add_special_tokens=False))
    per_image = max(1, prompt_tokens + approx_image_tokens)
    return max(1, min(MAX_BATCH_SIZE, batch_tokens // per_image))


def _is_vllm_available() -> bool:
    """Checks whether vLLM is installed."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _create_predictor(
    model_id: str,
    device: str,
    torch_dtype: str,
    use_flash_attention: bool,
    response_scheme,
    attrs_per_session: int | None,
    min_new_tokens: int,
    max_response_tokens: int | None,
    response_token_margin: int,
    debug: bool,
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    enable_thinking: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    temperature: float = 0.0,
):
    """Creates a VLM predictor instance.

    Handles auto-detection of thinking models from ``model_id``. The reasoning
    allowance is the predictor's own (``BaseVLMPredictor.single_call_budget``),
    not something added to ``max_response_tokens`` here, so that it also applies
    when the response budget is derived per session rather than given.

    Args:
        model_id: HuggingFace model ID.
        device: Device for HF backend.
        torch_dtype: Data type for HF backend.
        use_flash_attention: Flash attention for HF backend.
        response_scheme: ResponseScheme governing prompts and parsing. Passed in
            rather than rebuilt so the predictor cannot disagree with the dataset.
        attrs_per_session: Attributes per VLM session; None puts them all in one.
        min_new_tokens: Minimum tokens to generate.
        max_response_tokens: Maximum response tokens per session; None derives a
            per-session bound from the response scheme.
        response_token_margin: Absolute tokens added to a derived budget.
        debug: Enable debug output.
        backend: Backend to use ("auto", "hf", "vllm").
            - "auto": Use vLLM for FP8 models, HF otherwise
            - "hf": Force HuggingFace Transformers
            - "vllm": Force vLLM (required for FP8 models)
        gpu_memory_utilization: GPU memory fraction for vLLM (0.0-1.0).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        enable_thinking: Enable thinking/reasoning mode.  Auto-enabled when
            the model ID contains "thinking" (case-insensitive).
        thinking_budget: Tokens reserved for reasoning on top of the response
            budget, derived or explicit.

    Returns:
        A predictor instance with a `predict_batch()` method.
    """
    # Detect thinking model variants used without --enable-thinking
    if not enable_thinking and "thinking" in model_id.lower():
        raise ValueError(
            f"Model '{model_id}' is a thinking variant but --enable-thinking was not set. "
            "Pass enable_thinking=True (or --enable-thinking on the CLI) to use this model."
        )

    if enable_thinking:
        print(f"[INFO] Thinking enabled: {thinking_budget} reasoning tokens on top of the"
              f" response budget"
              + (f" of {max_response_tokens}" if max_response_tokens is not None
                 else ", which is derived per session from the response scheme"))

    # Determine model family
    is_fp8_model = "FP8" in model_id or "fp8" in model_id
    is_qwen3 = "Qwen3" in model_id
    model_id_lower = model_id.lower()
    is_gemma4 = "gemma-4" in model_id_lower or "gemma4" in model_id_lower

    if backend == "auto":
        if is_fp8_model:
            if _is_vllm_available():
                backend = "vllm"
                print(f"[INFO] Auto-selected vLLM backend for FP8 model: {model_id}")
            else:
                raise RuntimeError(
                    f"Model '{model_id}' is an FP8 model which requires vLLM, "
                    "but vLLM is not installed. Install with: pip install 'vllm>=0.11.0'"
                )
        else:
            backend = "hf"

    shared = dict(
        model_id=model_id,
        max_response_tokens=max_response_tokens,
        response_scheme=response_scheme,
        attrs_per_session=attrs_per_session,
        response_token_margin=response_token_margin,
        min_new_tokens=min_new_tokens,
        debug=debug,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        temperature=temperature,
    )

    if backend == "vllm":
        vllm_kwargs = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            **shared,
        )
        if is_gemma4:
            from vidlu_irap_gaim.vlm import Gemma4VLvLLMPredictor
            return Gemma4VLvLLMPredictor(**vllm_kwargs)
        from vidlu_irap_gaim.vlm import Qwen3VLvLLMPredictor
        return Qwen3VLvLLMPredictor(**vllm_kwargs)

    # HuggingFace backend
    if is_fp8_model:
        print(
            f"[WARNING] FP8 model '{model_id}' may not load correctly with HF backend. "
            "Consider using --backend vllm"
        )

    hf_kwargs = dict(
        device=device,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
        **shared,
    )
    if is_gemma4:
        from vidlu_irap_gaim.vlm import Gemma4VLPredictor
        return Gemma4VLPredictor(**hf_kwargs)
    if is_qwen3:
        from vidlu_irap_gaim.vlm import Qwen3VLPredictor
        return Qwen3VLPredictor(**hf_kwargs)
    raise ValueError(
        f"Unsupported model: '{model_id}'. "
        "Supported model families: Qwen3-VL (e.g. Qwen/Qwen3-VL-8B-Instruct), "
        "Gemma 4 (e.g. google/gemma-4-27b-it, google/gemma-4-31B-it)."
    )


def _session_records(
    segment_id: str,
    result,
    attr_to_position: dict[str, int],
    target,
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
    attrs_order: list[str],
) -> list[dict]:
    """One record per (segment, attribute), for the paired analysis.

    Not recoverable from ``predictions.json``: which response belonged to which
    question, how long it was, and whether its session ran out of budget are all
    per-session facts that a merged prediction dict has already thrown away.
    """
    attr_global_idx = {attr: i for i, attr in enumerate(attrs_order)}
    records = []
    for session_idx, session in enumerate(result.sessions):
        num_lines = sum(1 for line in session.response.splitlines() if line.strip())
        for attr in session.attrs:
            pred = session.predictions.get(attr)
            num_classes = len(attr_to_value_to_class_idx.get(attr, {}))
            target_idx = None
            if target is not None and attr in attr_global_idx:
                target_idx = int(target[attr_global_idx[attr]])
            records.append({
                "segment_id": segment_id,
                "attr": attr,
                # Where the attribute sits in the canonical order, so the
                # all-attributes arm can be checked for a position effect.
                "attr_position": attr_to_position.get(attr),
                "session_idx": session_idx,
                "num_attrs_in_session": len(session.attrs),
                "target_idx": target_idx,
                "pred_idx": None if pred is None else pred.pred_idx,
                "pred_value": None if pred is None else pred.pred_value,
                "valid": is_usable_prediction(pred, num_classes),
                "response": session.response,
                "num_response_lines": num_lines,
                "num_response_tokens": session.num_response_tokens,
                "is_truncated": session.is_truncated,
            })
    return records


def run_evaluation(
    *,
    # Dataset injection (optional - if provided, skips internal loading)
    dataset: Any = None,
    predictor: Any = None,
    # Standard parameters (used when dataset is not provided)
    dataset_name: str = "bih",
    split: str = "test",
    image_folder: str | None = None,
    output_dir: str | Path = "vlm_results",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    detail_level: str | None = None,
    response_scheme_name: str = "standard",
    limit: int | None = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    fail_fast: bool = True,
    attrs_per_session: int | None = None,
    min_new_tokens: int = 0,
    max_response_tokens: int | None = None,
    response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
    debug: bool = False,
    interactive: bool = True,
    print_prompt: bool = True,
    # Backend selection
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    batch_size: int | None = None,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
    enable_thinking: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    temperature: float = 0.0,
    upsampling_factor: int = 1,
) -> EvaluationResult:
    """Runs VLM evaluation on a dataset.

    Args:
        dataset: Optional pre-loaded VLM-wrapped dataset (skips internal loading).
            Its ``info`` supplies the response scheme and the attribute subset.
        predictor: Optional pre-loaded VLM predictor (skips model loading if provided).
            Useful for reusing the same model across multiple splits. Its
            ``attrs_per_session`` is left alone -- the caller configured it.
        split: Dataset split to evaluate ("train", "val", "test"). Ignored if dataset provided.
        image_folder: Optional custom image folder (overrides split). Ignored if dataset provided.
        output_dir: Directory to save results.
        model_id: HuggingFace model ID. Ignored if predictor provided.
        detail_level: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
        response_scheme_name: Response scheme for the dataset. Ignored if dataset provided.
        limit: Evaluate only the first N samples. A smoke-test knob, NOT a
            sample of the split: iRAP segments are stored grouped by road, so the
            first N come from a handful of roads.
        device: Device for inference. Ignored if predictor provided.
        torch_dtype: Data type for model. Ignored if predictor provided.
        use_flash_attention: Whether to use Flash Attention 2. Ignored if predictor provided.
        fail_fast: Stop on the first error (default True).
        attrs_per_session: Attributes per VLM session; None puts them all in one,
            1 gives each attribute its own prompt and its own session. Ignored if
            predictor provided.
        min_new_tokens: Minimum tokens to generate per session. Ignored if predictor provided.
        max_response_tokens: Maximum response tokens per session; None derives a
            per-session bound from the response scheme. Ignored if predictor provided.
        response_token_margin: Absolute tokens added to a derived budget.
        debug: Print detailed prompt/response debugging information.
        interactive: Enable interactive control (type "skip" to stop early). Default True.
        print_prompt: Print the first prompt sent to the VLM.
        backend: Backend for inference ("auto", "hf", "vllm"). Default "auto".
        gpu_memory_utilization: GPU memory fraction for vLLM backend (0.0-1.0).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        batch_size: Images generated for at once; None sizes it from ``batch_tokens``.
        batch_tokens: Target prompt tokens per batch, used when batch_size is None.
        temperature: Sampling temperature (0.0 = greedy). Ignored if predictor provided.

    Returns:
        EvaluationResult with summary statistics.
    """
    from vidlu_irap_gaim.vlm.predictions import predictions_to_json_serializable
    from vidlu.utils.misc import try_input

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        print(f"Loading dataset (split={split}, image_folder={image_folder})...")
        dataset = _load_dataset(split, image_folder, dataset_name, response_scheme_name,
                                detail_level, upsampling_factor)
        print(f"Dataset loaded: {len(dataset)} samples")
    else:
        print(f"Using provided dataset: {len(dataset)} samples")

    from vidlu_irap_gaim.vlm.finetuning.dataset import vlm_config_from_dataset

    config = vlm_config_from_dataset(dataset)
    response_scheme = config.response_scheme
    attrs_to_include = config.attrs_to_include
    attr_to_value_to_class_idx = config.attr_to_value_to_class_idx
    attrs_order = list(attr_to_value_to_class_idx.keys())
    attr_to_position = {attr: i + 1 for i, attr in enumerate(attrs_to_include)}
    # The dataset knows how verbose its prompts are; an explicit argument
    # overrides that, but a silent mismatch with training is not an option.
    if detail_level is None:
        detail_level = config.detail_level
    elif detail_level != config.detail_level:
        print(f"[WARNING] Prompting at detail_level={detail_level!r} while the dataset "
              f"was built with {config.detail_level!r}. Fine-tuned models were trained "
              f"against the latter.")

    has_labels = _dataset_has_labels(dataset)
    if has_labels:
        print("Labels available - will compute metrics")
        from vidlu_irap_gaim.metrics import get_irap_metrics

        # Two metric sets, one per scoring of an unusable response (see `vlm.scoring`): as
        # class 0, and excluded. The two response different questions, so both are reported.
        # Parsed text gives one-hot outputs, so no probabilistic metrics.
        metrics = get_irap_metrics(dataset, attrs_to_include=tuple(attrs_to_include),
                                   output_kind="hard")
        metrics_excluding_invalid = get_irap_metrics(dataset,
                                                     attrs_to_include=tuple(attrs_to_include),
                                                     output_kind="hard")
        for m in (metrics, metrics_excluding_invalid):
            m.reset()
    else:
        print("No labels - running prediction only")
        metrics = metrics_excluding_invalid = None

    missing_metadata = [a for a in attrs_to_include if a not in attr_to_value_to_class_idx]
    if missing_metadata:
        print(f"\n[CRITICAL] {len(missing_metadata)} attributes in eval list are MISSING "
              f"from dataset metadata:")
        for a in missing_metadata:
            print(f"  - {a}")
        print("Please check for naming mismatches or typos in attrs.py.")
        sys.exit(1)

    if predictor is None:
        print(f"Initializing VLM predictor (model={model_id}, backend={backend})...")
        print(f"  Attributes per session: "
              f"{'all' if attrs_per_session is None else attrs_per_session}")
        print(f"  Total attributes to evaluate: {len(attrs_to_include)}")
        print(f"  VLM sessions per sample: "
              f"{len(attribute_sessions(attrs_to_include, attrs_per_session))}")
        predictor = _create_predictor(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attention=use_flash_attention,
            response_scheme=response_scheme,
            attrs_per_session=attrs_per_session,
            min_new_tokens=min_new_tokens,
            max_response_tokens=max_response_tokens,
            response_token_margin=response_token_margin,
            debug=debug,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
        )
    else:
        print("Using provided predictor")
        model_id = getattr(predictor, "model_id", model_id)
        attrs_per_session = predictor.attrs_per_session
        # Hand over the dataset's convention rather than letting the predictor
        # fall back to its own default, which would prompt in one format while
        # the dataset scores in another. Raises on a genuine mismatch.
        predictor.response_scheme = response_scheme

    all_predictions = {}
    all_records: list[dict] = []
    num_valid = 0
    num_invalid_responses = 0
    num_scored_responses = 0
    num_truncated_sessions = 0
    num_sessions = 0
    num_samples_requested = min(len(dataset), limit) if limit else len(dataset)
    num_samples_completed = 0
    was_interrupted = False

    sessions = attribute_sessions(attrs_to_include, attrs_per_session)
    first_prompt = response_scheme.build_prompt(sessions[0], detail_level=detail_level)
    effective_batch_size = resolve_batch_size(predictor, first_prompt, batch_size, batch_tokens)

    print(f"Running evaluation on {num_samples_requested} samples "
          f"(batch_size={effective_batch_size})...")
    if interactive:
        print("(Type 'skip' to stop early and save partial results)")

    if print_prompt:
        print("\n" + "=" * 70)
        print("PROMPT (first session):")
        print("=" * 70)
        print(first_prompt)
        print("=" * 70 + "\n")

    started = time.perf_counter()
    for batch_start in tqdm(
        range(0, num_samples_requested, effective_batch_size),
        desc="VLM Inference",
        total=(num_samples_requested + effective_batch_size - 1) // effective_batch_size,
    ):
        if interactive:
            user_input = try_input()
            if user_input is not None and user_input.strip().lower() == "skip":
                print(f"\n[SKIP] Stopping early at batch starting at "
                      f"{batch_start}/{num_samples_requested}")
                was_interrupted = True
                break

        batch_end = min(batch_start + effective_batch_size, num_samples_requested)
        batch_indices = list(range(batch_start, batch_end))

        batch_samples = [dataset[idx] for idx in batch_indices]
        batch_images = [_extract_image(s) for s in batch_samples]
        batch_segment_ids = [_get_segment_id(s, idx)
                             for s, idx in zip(batch_samples, batch_indices)]

        try:
            batch_results = predictor.predict_batch(
                batch_images,
                attr_to_value_to_class_idx,
                attrs_to_include=attrs_to_include,
                detail_level=detail_level,
            )
        except Exception as e:
            if fail_fast:
                raise RuntimeError(
                    f"Error during VLM inference for batch {batch_start}-{batch_end}: {e}"
                ) from e
            for segment_id in batch_segment_ids:
                all_predictions[segment_id] = {"error": str(e)}
            num_samples_completed += len(batch_indices)
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            continue

        batch_predictions = [r.predictions for r in batch_results]
        batch_targets = None
        if has_labels:
            batch_targets = torch.stack([s["target"] for s in batch_samples])

        for sample, segment_id, result in zip(batch_samples, batch_segment_ids, batch_results):
            predictions = result.predictions

            valid_count = sum(1 for p in predictions.values() if p.pred_idx >= 0)
            if valid_count > 0:
                num_valid += 1

            all_predictions[segment_id] = {
                "predictions": predictions_to_json_serializable(predictions),
                "responses": result.responses,
            }

            all_records.extend(_session_records(
                segment_id, result, attr_to_position,
                sample["target"] if has_labels else None,
                attr_to_value_to_class_idx, attrs_order))
            num_scored, num_invalid = count_scored_and_invalid_responses(
                predictions, attrs_to_include, attr_to_value_to_class_idx)
            num_scored_responses += num_scored
            num_invalid_responses += num_invalid
            num_sessions += len(result.sessions)
            num_truncated_sessions += sum(1 for s in result.sessions if s.is_truncated)

            invalid_preds = {attr: pred for attr, pred in predictions.items()
                             if pred.pred_idx < 0}
            if invalid_preds:
                print(f"\n[INVALID PREDICTIONS] Sample {segment_id}")
                print(f"  Invalid attributes ({len(invalid_preds)}/{len(predictions)}):")
                for attr, pred in invalid_preds.items():
                    print(f"    - {attr}: predicted value '{pred.pred_value}' (no match found)")
                print("-" * 40)

            num_samples_completed += 1

        if has_labels:
            update_both_scorings(
                [metrics], [metrics_excluding_invalid], batch_predictions, batch_targets,
                attr_to_value_to_class_idx, attrs_order,
                attrs_to_include=attrs_to_include)

    duration_s = time.perf_counter() - started

    has_metrics = has_labels and num_samples_completed > 0
    # Built before the report so that what is printed and what is saved are read off the
    # returned object rather than recomputed from the same counters.
    result = EvaluationResult(
        num_samples_requested=num_samples_requested,
        num_samples_completed=num_samples_completed,
        num_valid_predictions=num_valid,
        metrics=metrics.compute() if has_metrics else None,
        metrics_excluding_invalid=(metrics_excluding_invalid.compute() if has_metrics
                                   else None),
        num_invalid_attribute_responses=num_invalid_responses,
        num_scored_attribute_responses=num_scored_responses,
        num_truncated_sessions=num_truncated_sessions,
        num_sessions=num_sessions,
        duration_s=duration_s,
        output_dir=output_dir,
        predictions_file=output_dir / "predictions.json",
        records_file=output_dir / "records.jsonl",
        was_interrupted=was_interrupted,
    )

    if has_metrics:
        print("\n=== Evaluation Metrics ===")
        if was_interrupted:
            print(f"  [WARNING] Metrics based on {num_samples_completed}/"
                  f"{num_samples_requested} samples (interrupted)")
        print_metrics("unusable responses scored as class 0", result.metrics)
        print_metrics("unusable responses excluded", result.metrics_excluding_invalid)

    print(f"\n  invalid responses: {num_invalid_responses}/{num_scored_responses} "
          f"({result.invalid_rate:.4f})")
    print(f"  truncated sessions: {num_truncated_sessions}/{num_sessions} "
          f"({result.truncation_rate:.4f})")
    if num_truncated_sessions:
        print("  [WARNING] A binding response budget in one arm of a comparison "
              "manufactures a difference between the arms. Raise "
              "response_token_margin and rerun before comparing.")

    with open(result.predictions_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    with open(result.records_file, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_samples_requested": num_samples_requested,
        "num_samples_completed": num_samples_completed,
        "num_valid_predictions": num_valid,
        "num_invalid_attribute_responses": num_invalid_responses,
        "num_scored_attribute_responses": num_scored_responses,
        "invalid_rate": result.invalid_rate,
        "num_truncated_sessions": num_truncated_sessions,
        "num_sessions": num_sessions,
        "truncation_rate": result.truncation_rate,
        "duration_s": duration_s,
        "was_interrupted": was_interrupted,
        "dataset": dataset_name,
        "split": split,
        "image_folder": str(image_folder) if image_folder else None,
        "model_id": model_id,
        "detail_level": detail_level,
        "attrs_per_session": attrs_per_session,
        "num_attrs": len(attrs_to_include),
        "batch_size": effective_batch_size,
        "metrics": None if result.metrics is None else metrics_to_json_dict(result.metrics),
        "metrics_excluding_invalid": (None if result.metrics_excluding_invalid is None
                                      else metrics_to_json_dict(result.metrics_excluding_invalid)),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - predictions.json: {len(all_predictions)} samples")
    print(f"  - records.jsonl: {len(all_records)} (segment, attribute) records")
    print("  - summary.json: evaluation summary")

    return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="VLM road attribute classification by generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        allow_abbrev=False,  # Require full argument names (disable prefix matching)
    )

    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--split", choices=["train", "val", "test"], default="test",
                            help="Dataset split to evaluate (default: test)")
    data_group.add_argument("--image-folder", type=str,
                            help="Custom folder of images to evaluate (overrides --split)")

    parser.add_argument("--dataset", choices=["bih", "vietnam"], default="bih",
                        help="Dataset to evaluate (default: bih)")
    parser.add_argument("--output-dir", type=str, default="vlm_results",
                        help="Output directory for results (default: vlm_results)")
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help=("HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct). "
              "Also supports Gemma 4: google/gemma-4-27b-it, google/gemma-4-31B-it"))
    parser.add_argument("--prompt-detail", dest="detail_level",
                        choices=["attr_desc_vals", "attr_vals", "attr", "none"],
                        default="attr_desc_vals",
                        help="Prompt detail level (default: attr_desc_vals)")
    parser.add_argument(
        "--response-scheme", dest="response_scheme_name", type=str, default="standard",
        help=("Prompt/response convention: standard (numbered lines), json, indexed, "
              "sparse_standard, sparse_indexed (default: standard). Governs prompts, "
              "ground truth and parsing alike."))
    parser.add_argument("--no-print-prompt", action="store_false", dest="print_prompt",
                        default=True,
                        help="Suppress printing the first prompt (enabled by default)")
    parser.add_argument(
        "--backend", type=str, choices=["auto", "hf", "vllm"], default="auto",
        help=("Backend for inference: 'auto' (vLLM for FP8, HF otherwise), "
              "'hf' (HuggingFace Transformers), 'vllm'. Default: auto"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="GPU memory fraction for vLLM backend (0.0-1.0, default: 0.80)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Number of GPUs for vLLM tensor parallelism (default: all)")
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help=("Max sequence length for vLLM KV cache allocation (default: 8192). "
              "Qwen3-VL models advertise very large max lengths (e.g. 262144), "
              "which cannot be allocated on a single 48 GB GPU."))
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=("Images generated for at once. Default: sized from --batch-tokens, "
              "which keeps the GPU equally busy for short and long prompts."))
    parser.add_argument(
        "--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS,
        help=f"Target prompt tokens per batch (default: {DEFAULT_BATCH_TOKENS})")

    parser.add_argument(
        "--limit", type=int,
        help=("Evaluate only the first N samples. A smoke-test knob, NOT a sample of "
              "the split: segments are stored grouped by road, so the first N come "
              "from a handful of roads."))
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model data type (default: bfloat16)")
    parser.add_argument("--no-flash-attention", action="store_true",
                        help="Disable Flash Attention 2")
    parser.add_argument("--no-fail-fast", action="store_true",
                        help="Continue after per-batch errors (NOT recommended)")
    parser.add_argument(
        "--attrs-per-session", type=int, default=None,
        help=("Attributes per VLM session, each session an independent (image, prompt) "
              "exchange. Default: all of them in one session. Pass 1 for one session "
              "and one prompt per attribute."))
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed prompt/response debugging information")
    parser.add_argument("--min-new-tokens", type=int, default=0,
                        help="Minimum tokens to generate per session (default: 0)")
    parser.add_argument(
        "--max-response-tokens", type=int, default=None,
        help=("Maximum response tokens per session. Default: derived per session from "
              "the response scheme, which knows the format and the closed value set, "
              "so the budget scales with how many attributes the session asks about."))
    parser.add_argument(
        "--response-token-margin", type=int, default=DEFAULT_RESPONSE_TOKEN_MARGIN,
        help=(f"Tokens added to a derived budget (default: {DEFAULT_RESPONSE_TOKEN_MARGIN}). "
              "The bound is there to "
              "catch gross non-compliance, not to leash the model: a budget that binds "
              "in one arm of a comparison manufactures a difference between the arms."))
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy decoding, default: 0.0)")
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help=("Enable thinking/reasoning mode. Adds --thinking-budget tokens on top of "
              "the response budget, whether that budget is derived or given explicitly. "
              "Auto-enabled when the model ID contains 'thinking' (case-insensitive)."))
    parser.add_argument(
        "--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET,
        help=(f"Tokens reserved for reasoning on top of the response budget "
              f"(default: {DEFAULT_THINKING_BUDGET})"))
    parser.add_argument("--upsampling-factor", type=int, default=1,
                        help="Upsample images by this integer factor (default: 1)")

    args = parser.parse_args()

    result = run_evaluation(
        dataset_name=args.dataset,
        split=args.split,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        model_id=args.model_id,
        detail_level=args.detail_level,
        response_scheme_name=args.response_scheme_name,
        limit=args.limit,
        device=args.device,
        torch_dtype=args.dtype,
        use_flash_attention=not args.no_flash_attention,
        fail_fast=not args.no_fail_fast,
        attrs_per_session=args.attrs_per_session,
        min_new_tokens=args.min_new_tokens,
        max_response_tokens=args.max_response_tokens,
        response_token_margin=args.response_token_margin,
        debug=args.debug,
        print_prompt=args.print_prompt,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        batch_tokens=args.batch_tokens,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
        upsampling_factor=args.upsampling_factor,
    )

    status = "INTERRUPTED" if result.was_interrupted else "complete"
    print(f"\nEvaluation {status}: {result.num_samples_completed}/"
          f"{result.num_samples_requested} samples, "
          f"{result.num_valid_predictions} valid predictions")
    if result.metrics:
        print(f"Metrics saved to {result.output_dir / 'summary.json'}")

    return 0


# =============================================================================
# Vidlu Experiment Entrypoint
# =============================================================================


def run(
    e,
    *,
    split_prefix: str = "test",
    output_subdir: str = "vlm_results",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    detail_level: str | None = None,
    limit: int | None = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    fail_fast: bool = True,
    attrs_per_session: int | None = None,
    min_new_tokens: int = 0,
    max_response_tokens: int | None = None,
    response_token_margin: int = DEFAULT_RESPONSE_TOKEN_MARGIN,
    debug: bool = False,
    interactive: bool = True,
    print_prompt: bool = True,
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    batch_size: int | None = None,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
    enable_thinking: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    temperature: float = 0.0,
):
    """Runs VLM evaluation using a Vidlu TrainingExperiment's datasets.

    The entrypoint for `scripts/run.py test -m irap_gaim.tools.vlm_inference`.
    It evaluates a *standalone* predictor (loaded from ``model_id``) on the
    experiment's prepared datasets. To evaluate the experiment's own model, see
    ``vidlu_irap_gaim.vlm.finetuning.predictor``.

    Args:
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate (e.g., "test", "val"). Default "test".
        output_subdir: Subdirectory under experiment dir for VLM results.
        attrs_per_session: Attributes per VLM session; None puts them all in one.
        (see ``run_evaluation`` for the rest)

    Returns:
        Dict mapping split names to EvaluationResult.
    """
    splits_to_eval = [(name, ds) for name, ds in e.data.items()
                      if name.startswith(split_prefix)]

    if not splits_to_eval:
        print(f"[WARNING] No splits found with prefix '{split_prefix}' in e.data")
        print(f"  Available splits: {list(e.data.keys())}")
        return {}

    print(f"VLM Evaluation on {len(splits_to_eval)} split(s): "
          f"{[name for name, _ in splits_to_eval]}")

    output_base = e.cpman.experiment_dir / output_subdir

    # The predictor needs the response scheme, which lives on the datasets.
    from vidlu_irap_gaim.vlm.finetuning.dataset import vlm_config_from_data
    response_scheme = vlm_config_from_data(dict(splits_to_eval)).response_scheme

    print(f"\nInitializing VLM predictor (model={model_id}, backend={backend})...")
    print(f"  Reused across all {len(splits_to_eval)} splits.")
    predictor = _create_predictor(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
        response_scheme=response_scheme,
        attrs_per_session=attrs_per_session,
        min_new_tokens=min_new_tokens,
        max_response_tokens=max_response_tokens,
        response_token_margin=response_token_margin,
        debug=debug,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        temperature=temperature,
    )

    results = {}
    for name, ds in splits_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating split: {name}")
        print(f"{'=' * 60}")

        results[name] = run_evaluation(
            dataset=ds,
            predictor=predictor,
            split=name,
            output_dir=output_base / name,
            model_id=model_id,
            detail_level=detail_level,
            limit=limit,
            fail_fast=fail_fast,
            debug=debug,
            interactive=interactive,
            print_prompt=print_prompt,
            batch_size=batch_size,
            batch_tokens=batch_tokens,
        )

    print(f"\n{'=' * 60}")
    print("VLM Evaluation Summary")
    print(f"{'=' * 60}")
    for name, result in results.items():
        status = "INTERRUPTED" if result.was_interrupted else "complete"
        print(f"  {name}: {status}, {result.num_samples_completed}/"
              f"{result.num_samples_requested} samples, "
              f"{result.num_valid_predictions} valid, "
              f"invalid_rate={result.invalid_rate:.4f}, "
              f"truncation_rate={result.truncation_rate:.4f}")
        if result.metrics:
            metric_strs = [f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:3]
                           if isinstance(v, (int, float))]
            if metric_strs:
                print(f"    Metrics: {', '.join(metric_strs)}")
    print(f"\nResults saved to: {output_base}")

    return results


if __name__ == "__main__":
    sys.exit(main())
