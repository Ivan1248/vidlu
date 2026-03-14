#!/usr/bin/env python3
"""
Benchmark vLLM inference time with and without prefix caching.

Each request has a different image and the same constant prompt. Tests caching
of the shared text prefix when images vary (typical VLM evaluation scenario).

Usage:
    python -m vidlu_irap_gaim.vlm.benchmark_prefix_caching
    python -m vidlu_irap_gaim.vlm.benchmark_prefix_caching --constant-len 11000 --target-output-chars 250

Requires: vllm, qwen-vl-utils, transformers, torch, PIL
"""

import argparse
import gc
import os
import time

import numpy as np
import torch
from PIL import Image


def _make_prompt(constant_len: int) -> str:
    """Build constant prompt of given character length."""
    filler = "This is the constant prefix for prefix caching. "
    return (filler * (constant_len // len(filler) + 1))[:constant_len]


def _make_dummy_image(
    width: int = 384,
    height: int = 288,
    seed: int | None = None,
) -> Image.Image:
    """Create a dummy PIL image for VLM input."""
    rng = np.random.default_rng(seed)
    arr = np.uint8(rng.random((height, width, 3)) * 255)
    return Image.fromarray(arr)


def _prepare_vllm_inputs(
    processor,
    constant_len: int,
    num_requests: int,
) -> list[dict]:
    """Prepare vLLM inputs: same constant prompt, different image per request."""
    from qwen_vl_utils import process_vision_info

    from vidlu_irap_gaim.vlm.qwen_utils import build_qwen_chat_messages

    prompt = _make_prompt(constant_len)
    inputs = []
    for i in range(num_requests):
        image = _make_dummy_image(seed=i)
        messages = build_qwen_chat_messages(image, prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        inputs.append({
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        })
    return inputs


def _run_benchmark(
    model_id: str,
    enable_prefix_caching: bool,
    num_requests: int,
    constant_len: int,
    target_output_chars: int,
    tensor_parallel_size: int | None,
) -> tuple[float, list[str]]:
    """Run inference benchmark and return total time and outputs."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print(f"\n  Loading model (enable_prefix_caching={enable_prefix_caching})...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tensor_parallel_size or torch.cuda.device_count(),
        max_model_len=8192,
        seed=42,
        enable_prefix_caching=enable_prefix_caching,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # ~4 chars per token; no floor so --target-output-chars is respected
    max_tokens = max(1, (target_output_chars // 4) + 1)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        top_k=-1,
    )

    vllm_inputs = _prepare_vllm_inputs(processor, constant_len, num_requests)

    print(f"  Running {num_requests} requests (constant prompt {constant_len} chars, variable image, max_tokens={max_tokens})...")
    start = time.perf_counter()
    outputs = llm.generate(vllm_inputs, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    texts = [o.outputs[0].text for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM inference with and without prefix caching."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests per run (default: 5)",
    )
    parser.add_argument(
        "--constant-len",
        type=int,
        default=11_000,
        help="Constant prompt length in characters (default: 11000)",
    )
    parser.add_argument(
        "--target-output-chars",
        type=int,
        default=1_500,
        help="Target output length in characters for max_tokens (default: 1500)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (default: all GPUs)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Prefix Caching Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Requests: {args.num_requests}")
    print(f"Prompt: constant {args.constant_len} chars, variable image per request")
    print(f"Target output: ~{args.target_output_chars} chars")

    print("\n--- Without prefix caching ---")
    time_no_cache, outputs_no = _run_benchmark(
        model_id=args.model,
        enable_prefix_caching=False,
        num_requests=args.num_requests,
        constant_len=args.constant_len,
        target_output_chars=args.target_output_chars,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print(f"  Total time: {time_no_cache:.2f} s")
    print(f"  Per request: {time_no_cache / args.num_requests:.2f} s")
    print(f"  Output lengths: {[len(t) for t in outputs_no]}")

    print("\n--- With prefix caching ---")
    time_with_cache, outputs_with = _run_benchmark(
        model_id=args.model,
        enable_prefix_caching=True,
        num_requests=args.num_requests,
        constant_len=args.constant_len,
        target_output_chars=args.target_output_chars,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print(f"  Total time: {time_with_cache:.2f} s")
    print(f"  Per request: {time_with_cache / args.num_requests:.2f} s")
    print(f"  Output lengths: {[len(t) for t in outputs_with]}")

    print("\n--- Summary ---")
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0
    print(f"  Without cache: {time_no_cache:.2f} s")
    print(f"  With cache:    {time_with_cache:.2f} s")
    print(f"  Speedup:       {speedup:.2f}x")


if __name__ == "__main__":
    main()
