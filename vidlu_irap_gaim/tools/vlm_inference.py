"""
Standalone VLM-based zero-shot road attribute classification evaluation.

This module provides a command-line interface for running VLM evaluation
on the BiH dataset or custom image folders.

Usage (CLI):
    # Evaluate on test split with ground truth
    IRAP_HOME=/path/to/data python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --output-dir results/vlm_zeroshot

    # Evaluate on custom folder (no labels)
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --image-folder /path/to/images --output-dir results/vlm_custom

    # Use more verbose prompts for better accuracy
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --prompt-detail attr_desc_vals

    # Use FP8 model with vLLM backend (auto-selected for FP8 models)
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --model-id "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

    # Explicitly select backend
    python -m vidlu_irap_gaim.tools.vlm_inference \
        --split test --backend vllm --model-id "Qwen/Qwen3-VL-8B-Instruct"

Usage (via Vidlu runner):
    # Evaluate using experiment's prepared datasets
    python scripts/run.py test <data> <input_adapter> <model> <trainer> \\
        --params <params> -m irap_gaim.tools.vlm_inference

    # Pass VLM-specific arguments
    python scripts/run.py test ... -m irap_gaim.tools.vlm_inference:run,model_id="Qwen/Qwen3-VL-8B-Instruct"

    # Use FP8 model with vLLM
    python scripts/run.py test ... -m irap_gaim.tools.vlm_inference:run,model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

Interactive control:
    Type "skip" during evaluation to stop early and save partial results.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

# Lazy imports to avoid loading heavy dependencies at import time


@dataclass
class EvaluationResult:
    """Summary of VLM evaluation results."""

    num_samples_requested: int
    num_samples_completed: int
    num_valid_predictions: int
    metrics: dict[str, float] | None
    output_dir: Path
    predictions_file: Path
    was_interrupted: bool = False


def _load_dataset(
    split: str,
    image_folder: str | None,
) -> tuple[Any, dict[str, dict[str, int]], list[str]]:
    """Load dataset for evaluation.

    Returns:
        Tuple of (dataset, attr_to_value_to_class_idx, attrs_order)
    """
    from vidlu_irap_gaim.datasets import make_bih_data, InferenceImageDataset
    from vidlu_irap_gaim.attrs import get_attrs_to_include

    if image_folder is not None:
        # Custom folder - need reference dataset for metadata
        ref_data = make_bih_data()
        ref_ds = ref_data["test"]
        dataset = InferenceImageDataset.from_folder(
            image_folder,
            reference_dataset=ref_ds,
        )
        attr_to_value_to_class_idx = ref_ds.info.attr_to_value_to_class_idx
    else:
        # Standard split
        data = make_bih_data()
        dataset = data[split]
        attr_to_value_to_class_idx = dataset.info.attr_to_value_to_class_idx

    attrs_order = list(attr_to_value_to_class_idx.keys())
    attrs_to_include = list(get_attrs_to_include())

    return dataset, attr_to_value_to_class_idx, attrs_order, attrs_to_include


def _create_metrics(
    dataset: Any,
    attrs_to_include: list[str],
) -> tuple[Any, ...] | None:
    """Create metrics for evaluation (only if dataset has labels)."""
    # Check if dataset has labels
    try:
        sample = dataset[0]
        has_labels = hasattr(sample, "keys") and "target" in sample.keys()
    except Exception:
        has_labels = False

    if not has_labels:
        return None

    from vidlu_irap_gaim.metrics import get_irap_metrics

    return get_irap_metrics(dataset, attrs_to_include=tuple(attrs_to_include))


def _is_vllm_available() -> bool:
    """Check if vLLM is installed."""
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
    prompt_config: str | None,
    chunk_size: int,
    min_new_tokens: int,
    max_new_tokens: int,
    debug: bool,
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
):
    """Create a VLM predictor instance.

    Args:
        model_id: HuggingFace model ID.
        device: Device for HF backend.
        torch_dtype: Data type for HF backend.
        use_flash_attention: Flash attention for HF backend.
        prompt_config: Path to YAML prompt config.
        chunk_size: Max attributes per VLM call.
        min_new_tokens: Minimum tokens to generate.
        max_new_tokens: Maximum tokens to generate.
        debug: Enable debug output.
        backend: Backend to use ("auto", "hf", "vllm").
            - "auto": Use vLLM for FP8 models, HF otherwise
            - "hf": Force HuggingFace Transformers
            - "vllm": Force vLLM (required for FP8 models)
        gpu_memory_utilization: GPU memory fraction for vLLM (0.0-1.0).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.

    Returns:
        A predictor instance with a `predict()` method.
    """
    from vidlu_irap_gaim.vlm import (
        Qwen3VLPredictor,
        Qwen3VLvLLMPredictor,
    )

    # Determine backend
    is_fp8_model = "FP8" in model_id or "fp8" in model_id
    is_qwen3 = "Qwen3" in model_id

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

    if backend == "vllm":
        from vidlu_irap_gaim.vlm import Qwen3VLvLLMPredictor

        return Qwen3VLvLLMPredictor(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            prompt_config_path=prompt_config,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            debug=debug,
        )
    else:
        # HuggingFace backend
        if is_fp8_model:
            print(
                f"[WARNING] FP8 model '{model_id}' may not load correctly with HF backend. "
                "Consider using --backend vllm"
            )

        if not is_qwen3:
            raise ValueError(
                f"Model '{model_id}' is not Qwen3-VL. "
                "Qwen2.5-VL has been removed. Use a Qwen3-VL model "
                "(e.g. Qwen/Qwen3-VL-8B-Instruct) or --backend vllm for Qwen3-VL."
            )

        return Qwen3VLPredictor(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attention=use_flash_attention,
            prompt_config_path=prompt_config,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            debug=debug,
        )


def run_evaluation(
    *,
    # Dataset injection (optional - if provided, skips internal loading)
    dataset: Any = None,
    predictor: Any = None,
    # Standard parameters (used when dataset is not provided)
    split: str = "test",
    image_folder: str | None = None,
    output_dir: str | Path = "vlm_results",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    prompt_detail: str = "attr_desc_vals",
    output_format: str = "standard",
    prompt_config: str | None = None,
    limit: int | None = None,
    save_visualizations: bool = False,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    fail_fast: bool = True,
    allow_invalid_predictions: bool = False,
    chunk_size: int = 10,
    min_new_tokens: int = 0,
    max_new_tokens: int = 512,
    debug: bool = False,
    interactive: bool = True,
    print_prompt: bool = False,
    # Backend selection
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    batch_size: int = 1,
) -> EvaluationResult:
    """Run VLM evaluation on a dataset.

    Args:
        dataset: Optional pre-loaded dataset (skips internal loading if provided).
            Must have `info.attr_to_value_to_class_idx` attribute.
        predictor: Optional pre-loaded VLM predictor (skips model loading if provided).
            Useful for reusing the same model across multiple splits.
        split: Dataset split to evaluate ("train", "val", "test"). Ignored if dataset provided.
        image_folder: Optional custom image folder (overrides split). Ignored if dataset provided.
        output_dir: Directory to save results.
        model_id: HuggingFace model ID. Ignored if predictor provided.
        prompt_detail: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
        output_format: VLM output format ("json" for verbose, "standard" for numbered lines).
        prompt_config: Optional path to custom prompt YAML config. Ignored if predictor provided.
        limit: Maximum number of samples to evaluate.
        save_visualizations: Whether to save visualization images.
        device: Device for inference. Ignored if predictor provided.
        torch_dtype: Data type for model. Ignored if predictor provided.
        use_flash_attention: Whether to use Flash Attention 2. Ignored if predictor provided.
        fail_fast: Stop on first error (default True).
        allow_invalid_predictions: Map invalid predictions to class 0 (biases metrics).
        chunk_size: Max attributes per VLM call. Ignored if predictor provided.
        min_new_tokens: Minimum tokens to generate per VLM call. Ignored if predictor provided.
        max_new_tokens: Maximum tokens to generate per VLM call. Ignored if predictor provided.
        debug: Print detailed prompt/response debugging information.
        interactive: Enable interactive control (type "skip" to stop early). Default True.
        print_prompt: Print the first prompt sent to the VLM (for debugging).
        backend: Backend for inference ("auto", "hf", "vllm"). Default "auto".
            - "auto": Use vLLM for FP8 models, HF otherwise
            - "hf": Force HuggingFace Transformers
            - "vllm": Force vLLM (required for FP8 models)
        gpu_memory_utilization: GPU memory fraction for vLLM backend (0.0-1.0).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        batch_size: Number of images to process per batch. vLLM benefits from larger
            batches (e.g., 8). Default 1 for sequential compatibility.

    Returns:
        EvaluationResult with summary statistics.
    """
    from vidlu_irap_gaim.vlm import (
        extract_center_frame,
        predictions_to_output_tuple,
        predictions_to_json_serializable,
    )
    from vidlu.utils.collections import NameDict
    from vidlu.utils.misc import try_input
    from vidlu_irap_gaim.attrs import get_attrs_to_include

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset if not provided
    if dataset is None:
        print(f"Loading dataset (split={split}, image_folder={image_folder})...")
        dataset, attr_to_value_to_class_idx, attrs_order, attrs_to_include = _load_dataset(
            split, image_folder
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    else:
        # Extract metadata from provided dataset
        print(f"Using provided dataset: {len(dataset)} samples")
        attr_to_value_to_class_idx = dataset.info.attr_to_value_to_class_idx
        attrs_order = list(attr_to_value_to_class_idx.keys())
        attrs_to_include = list(get_attrs_to_include())

    # Create metrics if dataset has labels
    metrics = _create_metrics(dataset, attrs_to_include)
    has_labels = metrics is not None
    if has_labels:
        print("Labels available - will compute metrics")
        for m in metrics:
            m.reset()
    else:
        print("No labels - running prediction only")

    # Determine prompt config path
    if prompt_config is None:
        # Use default config from vlm package
        default_config = Path(__file__).parent.parent / "vlm" / "attribute_prompts.yaml"
        if default_config.exists():
            prompt_config = str(default_config)

    # Validate that all evaluation attributes exist in the dataset metadata
    missing_metadata = [a for a in attrs_to_include if a not in attr_to_value_to_class_idx]
    if missing_metadata:
        print(f"\n[CRITICAL] {len(missing_metadata)} attributes in eval list are MISSING from dataset metadata:")
        for a in missing_metadata:
            print(f"  - {a}")
        print("Please check for naming mismatches or typos in attrs.py.")
        sys.exit(1)

    # Initialize VLM predictor if not provided
    if predictor is None:
        print(f"Initializing VLM predictor (model={model_id}, backend={backend})...")
        print(f"  Chunk size: {chunk_size} attributes per VLM call")
        print(f"  Total attributes to evaluate: {len(attrs_to_include)}")
        print(f"  Estimated VLM calls per sample: {(len(attrs_to_include) + chunk_size - 1) // chunk_size}")
        predictor = _create_predictor(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            use_flash_attention=use_flash_attention,
            prompt_config=prompt_config,
            chunk_size=chunk_size,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            debug=debug,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
    else:
        print("Using provided predictor")
        model_id = getattr(predictor, "model_id", model_id)

    # Run evaluation
    all_predictions = {}
    num_valid = 0
    num_samples_requested = min(len(dataset), limit) if limit else len(dataset)
    num_samples_completed = 0
    was_interrupted = False

    # Helper to extract image from sample
    def extract_image_from_sample(sample):
        rgb = sample["rgb"] if hasattr(sample, "keys") else sample[0]
        if rgb.ndim == 4:  # (S, C, H, W) sequence
            return extract_center_frame(rgb, frame_index=0)
        else:  # (C, H, W) single image
            return Image.fromarray(
                (rgb.permute(1, 2, 0).numpy() * 255).astype("uint8")
            )

    # Helper to get segment ID from sample
    def get_segment_id(sample, idx):
        return sample.get("segment_id", str(idx)) if hasattr(sample, "get") else str(idx)

    effective_batch_size = batch_size
    print(f"Running evaluation on {num_samples_requested} samples (batch_size={effective_batch_size})...")
    print(f"  Output format: {output_format}")
    if interactive:
        print("(Type 'skip' to stop early and save partial results)")

    prompt_printed = False  # Track whether we've printed the prompt

    # Process samples in batches
    for batch_start in tqdm(
        range(0, num_samples_requested, effective_batch_size),
        desc="VLM Inference",
        total=(num_samples_requested + effective_batch_size - 1) // effective_batch_size,
    ):
        # Check for interactive skip command
        if interactive:
            user_input = try_input()
            if user_input is not None and user_input.strip().lower() == "skip":
                print(f"\n[SKIP] Stopping early at batch starting at {batch_start}/{num_samples_requested}")
                was_interrupted = True
                break

        batch_end = min(batch_start + effective_batch_size, num_samples_requested)
        batch_indices = list(range(batch_start, batch_end))

        # Collect batch samples
        batch_samples = [dataset[idx] for idx in batch_indices]
        batch_images = [extract_image_from_sample(s) for s in batch_samples]
        batch_segment_ids = [get_segment_id(s, idx) for s, idx in zip(batch_samples, batch_indices)]

        # Run batched VLM prediction
        try:
            batch_results = predictor.predict_batch(
                batch_images,
                attr_to_value_to_class_idx,
                attrs_to_include=attrs_to_include,
                detail_level=prompt_detail,
                output_format=output_format,
            )

            # Print the first prompt if requested (once only)
            if print_prompt and not prompt_printed and batch_results:
                first_result = batch_results[0]
                prompt_text = first_result.prompt
                # If chunked, just show the first chunk
                if first_result.chunk_prompts:
                    prompt_text = first_result.chunk_prompts[0]
                print("\n" + "=" * 70)
                print("PROMPT (first sample, first chunk):")
                print("=" * 70)
                print(prompt_text)
                print("=" * 70 + "\n")
                prompt_printed = True

        except Exception as e:
            if fail_fast:
                raise RuntimeError(
                    f"Error during VLM inference for batch {batch_start}-{batch_end}: {e}"
                ) from e
            # On error, mark all samples in batch as failed
            for segment_id in batch_segment_ids:
                all_predictions[segment_id] = {"error": str(e)}
            num_samples_completed += len(batch_indices)
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            continue

        # Process each result in the batch
        for sample, segment_id, result in zip(batch_samples, batch_segment_ids, batch_results):
            predictions = result.predictions

            # Count valid predictions
            valid_count = sum(1 for p in predictions.values() if p.pred_idx >= 0)
            if valid_count > 0:
                num_valid += 1

            # Store predictions
            all_predictions[segment_id] = {
                "predictions": predictions_to_json_serializable(predictions),
                "raw_response": result.raw_response,
            }

            # Update metrics if labels available
            if has_labels:
                target = sample["target"]
                if isinstance(target, torch.Tensor):
                    target = target.unsqueeze(0)  # Add batch dim

                try:
                    out = predictions_to_output_tuple(
                        predictions,
                        attr_to_value_to_class_idx,
                        attrs_order,
                        batch_size=1,
                        device="cpu",
                        allow_invalid=allow_invalid_predictions,
                        required_attrs=set(attrs_to_include),
                    )
                except ValueError as e:
                    print(f"\n[FAIL-FAST] Attribute conversion failed for sample {segment_id}")
                    print(f"Error: {e}")
                    print(f"VLM Raw Response:\n{result.raw_response}")
                    print("-" * 40)
                    if fail_fast:
                        raise
                    num_samples_completed += 1
                    continue

                iter_result = NameDict(out=out, target=target)
                for m in metrics:
                    m.update(iter_result)

            # Print invalid predictions (even when allow_invalid_predictions is True)
            invalid_preds = {
                attr: pred for attr, pred in predictions.items()
                if pred.pred_idx < 0
            }
            if invalid_preds:
                print(f"\n[INVALID PREDICTIONS] Sample {segment_id}")
                print(f"  Invalid attributes ({len(invalid_preds)}/{len(predictions)}):")
                for attr, pred in invalid_preds.items():
                    print(f"    - {attr}: predicted value '{pred.pred_value}' (no match found)")
                print(f"  VLM Raw Response:\n{result.raw_response}")
                print("-" * 40)

            num_samples_completed += 1

    # Compute final metrics
    computed_metrics = None
    if has_labels and metrics and num_samples_completed > 0:
        computed_metrics = {}
        for m in metrics:
            computed_metrics.update(m.compute())
        print("\n=== Evaluation Metrics ===")
        if was_interrupted:
            print(f"  [WARNING] Metrics based on {num_samples_completed}/{num_samples_requested} samples (interrupted)")
        for k, v in computed_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                # Per-attribute metrics
                avg = sum(v.values()) / len(v) if v else 0
                print(f"  {k} (avg): {avg:.4f}")

    # Save results (always save, even if interrupted)
    predictions_file = output_dir / "predictions.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    summary = {
        "num_samples_requested": num_samples_requested,
        "num_samples_completed": num_samples_completed,
        "num_valid_predictions": num_valid,
        "was_interrupted": was_interrupted,
        "split": split,
        "image_folder": str(image_folder) if image_folder else None,
        "model_id": model_id,
        "prompt_detail": prompt_detail,
        "metrics": computed_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - predictions.json: {len(all_predictions)} samples")
    print("  - summary.json: evaluation summary")

    return EvaluationResult(
        num_samples_requested=num_samples_requested,
        num_samples_completed=num_samples_completed,
        num_valid_predictions=num_valid,
        metrics=computed_metrics,
        output_dir=output_dir,
        predictions_file=predictions_file,
        was_interrupted=was_interrupted,
    )


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="VLM-based zero-shot road attribute classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        allow_abbrev=False,  # Require full argument names (disable prefix matching)
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    data_group.add_argument(
        "--image-folder",
        type=str,
        help="Custom folder of images to evaluate (overrides --split)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vlm_results",
        help="Output directory for results (default: vlm_results)",
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--prompt-detail",
        choices=["attr_desc_vals", "attr_vals", "attr", "none"],
        default="attr_desc_vals",
        help="Prompt detail level (default: attr_desc_vals)",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        help="Custom prompt YAML configuration file",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "standard"],
        default="standard",
        help=(
            "VLM output format: 'json' (verbose with full attribute names) or "
            "'standard' (numbered lines like '1: None'). Standard saves ~5x output tokens. "
            "Default: standard"
        ),
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the first prompt sent to the VLM (for debugging)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "hf", "vllm"],
        default="auto",
        help=(
            "Backend for inference: 'auto' (vLLM for FP8, HF otherwise), "
            "'hf' (HuggingFace Transformers), 'vllm' (vLLM - required for FP8 models). "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
        help="GPU memory fraction for vLLM backend (0.0-1.0, default: 0.80)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs for vLLM tensor parallelism (default: all available)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help=(
            "Max sequence length for vLLM KV cache allocation (default: 8192). "
            "Qwen3-VL models advertise very large max lengths (e.g. 262144), "
            "which cannot be allocated on a single 48 GB GPU."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for inference (default: 1). vLLM benefits significantly "
            "from larger batches (e.g., 8). Auto-set to 8 for vLLM backend if not specified."
        ),
    )

    # Inference options
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model data type (default: bfloat16)",
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue after per-sample errors (NOT recommended for evaluation)",
    )
    parser.add_argument(
        "--allow-invalid-predictions",
        action="store_true",
        help="Allow invalid/unmatched predictions by mapping them to class 0 (will bias metrics)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=52,
        help="Max attributes per VLM call to avoid context overflow (default: 10)",
    )
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="Evaluate attributes one by one (equivalent to --chunk-size 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed prompt/response debugging information",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum tokens to generate per VLM call. If >0, may force extra junk after JSON (default: 0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per VLM call (default: 512)",
    )

    args = parser.parse_args()

    if args.one_by_one:
        args.chunk_size = 1

    # Auto-select batch size for vLLM if user didn't specify
    effective_batch_size = args.batch_size
    if args.backend == "vllm" or (args.backend == "auto" and ("FP8" in args.model_id or "fp8" in args.model_id)):
        if args.batch_size == 1:
            effective_batch_size = 8
            print(f"[INFO] Auto-set batch_size={effective_batch_size} for vLLM efficiency")

    result = run_evaluation(
        split=args.split,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        model_id=args.model_id,
        prompt_detail=args.prompt_detail,
        output_format=args.output_format,
        prompt_config=args.prompt_config,
        limit=args.limit,
        device=args.device,
        torch_dtype=args.dtype,
        use_flash_attention=not args.no_flash_attention,
        fail_fast=not args.no_fail_fast,
        allow_invalid_predictions=args.allow_invalid_predictions,
        chunk_size=args.chunk_size,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        debug=args.debug,
        print_prompt=args.print_prompt,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=effective_batch_size,
    )

    status = "INTERRUPTED" if result.was_interrupted else "complete"
    print(f"\nEvaluation {status}: {result.num_samples_completed}/{result.num_samples_requested} samples, "
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
    prompt_detail: str = "attr_desc_vals",
    output_format: str = "standard",
    prompt_config: str | None = None,
    limit: int | None = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    fail_fast: bool = True,
    allow_invalid_predictions: bool = False,
    chunk_size: int = 10,
    min_new_tokens: int = 0,
    max_new_tokens: int = 512,
    debug: bool = False,
    interactive: bool = True,
    print_prompt: bool = False,
    # Backend selection
    backend: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int | None = None,
    max_model_len: int = 8192,
    batch_size: int = 1,
):
    """Run VLM evaluation using a Vidlu TrainingExperiment's datasets.

    This is the entrypoint for `scripts/run.py test -m irap_gaim.tools.vlm_inference`.
    It evaluates on the experiment's prepared datasets (e.data) and saves results
    under the experiment's checkpoint directory.

    Args:
        e: TrainingExperiment instance from Vidlu.
        split_prefix: Prefix for splits to evaluate (e.g., "test", "val"). Default "test".
        output_subdir: Subdirectory under experiment dir for VLM results. Default "vlm_results".
        model_id: HuggingFace model ID for VLM.
        prompt_detail: Prompt detail level ("attr_desc_vals", "attr_vals", "attr", "none").
        output_format: VLM output format ("json" or "standard").
        prompt_config: Optional path to custom prompt YAML config.
        limit: Maximum number of samples per split.
        device: Device for inference (HF backend only).
        torch_dtype: Data type for model (HF backend only).
        use_flash_attention: Whether to use Flash Attention 2 (HF backend only).
        fail_fast: Stop on first error (default True).
        allow_invalid_predictions: Map invalid predictions to class 0.
        chunk_size: Max attributes per VLM call.
        min_new_tokens: Minimum tokens to generate per VLM call.
        max_new_tokens: Maximum tokens to generate per VLM call.
        debug: Print detailed prompt/response debugging information.
        interactive: Enable interactive control (type "skip" to stop early).
        print_prompt: Print the first prompt sent to the VLM.
        backend: Backend for inference ("auto", "hf", "vllm"). Default "auto".
        gpu_memory_utilization: GPU memory fraction for vLLM (0.0-1.0).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        batch_size: Number of images to process per batch (default 1, auto-set to 8 for vLLM).

    Returns:
        Dict mapping split names to EvaluationResult.
    """
    # Auto-select batch size for vLLM if user didn't specify
    effective_batch_size = batch_size
    is_fp8_model = "FP8" in model_id or "fp8" in model_id
    if backend == "vllm" or (backend == "auto" and is_fp8_model):
        if batch_size == 1:
            effective_batch_size = 8
            print(f"[INFO] Auto-set batch_size={effective_batch_size} for vLLM efficiency")

    # Select splits to evaluate
    splits_to_eval = [
        (name, ds)
        for name, ds in e.data.items()
        if name.startswith(split_prefix)
    ]

    if not splits_to_eval:
        print(f"[WARNING] No splits found with prefix '{split_prefix}' in e.data")
        print(f"  Available splits: {list(e.data.keys())}")
        return {}

    print(f"VLM Evaluation on {len(splits_to_eval)} split(s): {[name for name, _ in splits_to_eval]}")

    # Determine output base directory
    output_base = e.cpman.experiment_dir / output_subdir

    # Create predictor once (expensive operation)
    print(f"\nInitializing VLM predictor (model={model_id}, backend={backend})...")
    print(f"  This will be reused across all {len(splits_to_eval)} splits.")
    predictor = _create_predictor(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
        prompt_config=prompt_config,
        chunk_size=chunk_size,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        debug=debug,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # Evaluate each split
    results = {}
    for name, ds in splits_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating split: {name}")
        print(f"{'=' * 60}")

        split_output_dir = output_base / name
        result = run_evaluation(
            dataset=ds,
            predictor=predictor,
            split=name,
            output_dir=split_output_dir,
            model_id=model_id,
            prompt_detail=prompt_detail,
            output_format=output_format,
            prompt_config=prompt_config,
            limit=limit,
            fail_fast=fail_fast,
            allow_invalid_predictions=allow_invalid_predictions,
            debug=debug,
            interactive=interactive,
            print_prompt=print_prompt,
            batch_size=effective_batch_size,
        )
        results[name] = result

    # Print summary
    print(f"\n{'=' * 60}")
    print("VLM Evaluation Summary")
    print(f"{'=' * 60}")
    for name, result in results.items():
        status = "INTERRUPTED" if result.was_interrupted else "complete"
        samples_info = f"{result.num_samples_completed}/{result.num_samples_requested}"
        print(f"  {name}: {status}, {samples_info} samples, {result.num_valid_predictions} valid")
        if result.metrics:
            # Show first few metrics
            metric_strs = [f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:3]
                          if isinstance(v, (int, float))]
            if metric_strs:
                print(f"    Metrics: {', '.join(metric_strs)}")
    print(f"\nResults saved to: {output_base}")

    return results


if __name__ == "__main__":
    sys.exit(main())
