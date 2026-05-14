"""
Qwen3.6-VL predictor using vLLM for zero-shot road attribute classification.

BF16 is used by default because RTX A6000 (Ampere) has no native FP8
compute support; BF16 avoids the Marlin fallback penalty and fits
comfortably across 4×49 GB GPUs (~18 GB/GPU for weights).
Pass ``model_id="Qwen/Qwen3.6-35B-A3B-FP8"`` explicitly to use the FP8
checkpoint, e.g. when a larger ``max_model_len`` is needed.

Qwen3.6 always thinks by default (``<think>...</think>`` blocks).
Use ``enable_thinking=True`` to retain the thinking text in results.
Note: Qwen3.6 does not support the ``/think`` / ``/nothink`` soft-switch
of Qwen3; thinking is controlled via the chat template's
``enable_thinking`` parameter, which this class sets automatically.

The ``temperature=0.0`` default is kept for deterministic classification.
For general-purpose use the Qwen3.6 team recommends:
  - Non-thinking: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
  - Thinking (general): temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
  - Thinking (coding): temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
"""

from functools import partial

from .qwen3_vl_vllm import Qwen3VLvLLMPredictor

Qwen36VLvLLMPredictor = partial(
    Qwen3VLvLLMPredictor,
    model_id="Qwen/Qwen3.6-35B-A3B",
    max_model_len=32768,
)
