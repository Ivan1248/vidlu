"""
Multi-GPU 1-step smoke test for Gemma4VLClassifier fine-tuning.

Why multi-GPU: bnb 4-bit silently fails to quantize Gemma 4's MoE expert
linears (unslothai/unsloth#4907) — load consumes ~48 GB on a single A6000.
The supported path on 4× A6000 is bf16 base + ``device_map="auto"`` (naive
model parallel; ~12.5 GB of weights per GPU).  This script uses the
``Gemma4VLClassifier`` class defaults (bf16, device_map="auto") unless the
flags below force otherwise.

Goal: confirm that the chosen model + library versions can:
  1. Load google/gemma-4-26B-A4B-it in bf16 across visible GPUs.
  2. Attach a LoRA adapter on the default regex
     (``.*language_model.*.{q,k,v,o,gate,up,down,gate_up}_proj``) and
     produce a non-trivial trainable-param count.
  3. Run one forward + backward on a synthetic batch.
  4. Stay under per-GPU VRAM budget.

Expected outcomes:
  - Loads cleanly.  Trainable-param fraction in the low single-digit % range
    (significantly higher than the attention-only baseline of ~0.3 %).
  - Loss is finite.  At least one LoRA param has non-zero gradient.
  - Per-GPU peak VRAM well under 48 GB on a 4× A6000 node.

Usage (4× A6000 node):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/sanity_gemma4_qlora.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image  # noqa: F401  (kept for clarity; used transitively via _to_pil_image)


def make_dummy_batch(batch_size: int, image_size: int) -> dict:
    """Synthetic batch shaped like what VLMTrainStep would see."""
    images = torch.randint(0, 256, (batch_size, 3, image_size, image_size), dtype=torch.uint8)
    prompts = ["Describe what's in this image."] * batch_size
    responses = ["A short test response."] * batch_size
    targets = torch.zeros(batch_size, 1, dtype=torch.long)
    return {
        "image": images,
        "prompt": prompts,
        "response": responses,
        "target": targets,
    }


def print_per_gpu_memory(label: str) -> None:
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"[sanity] {label} GPU{i}: alloc={alloc:.2f} GB reserved={reserved:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model id (default: %(default)s).",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force NF4 quantization (default: use class default — False for Gemma 4 "
             "because bnb 4-bit silently skips its MoE expert linears).",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Override HF device_map (default: class default — 'auto' for Gemma 4). "
             "Pass 'single' to force {\"\": 'cuda'}.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=None,
        help="Override LoRA target modules as a space-separated list of suffix matches. "
             "Default is the model-class default.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[sanity] CUDA is not available; aborting.", file=sys.stderr)
        return 2

    n_gpus = torch.cuda.device_count()
    print(f"[sanity] {n_gpus} CUDA device(s) visible.")

    # Lazy import so a missing transformers version produces a clear error.
    from vidlu_irap_gaim.vlm.finetuning import Gemma4VLClassifier  # noqa: E402

    device_map_override = None
    if args.device_map == "single":
        device_map_override = {"": "cuda"}
    elif args.device_map:
        device_map_override = args.device_map

    cls_kwargs = dict(
        model_id=args.model_id,
        lora_r=args.lora_r,
        lora_target_modules=tuple(args.lora_targets) if args.lora_targets else None,
        use_gradient_checkpointing=True,
    )
    if args.load_in_4bit:
        cls_kwargs["load_in_4bit"] = True
    if device_map_override is not None:
        cls_kwargs["device_map"] = device_map_override

    classifier = Gemma4VLClassifier(**cls_kwargs)
    print(f"[sanity] load_in_4bit={classifier.load_in_4bit}, "
          f"device_map={classifier.device_map}")

    # Only move the placeholder if we're definitely single-device.  With
    # device_map="auto", the real model is sharded by HF, and we leave it.
    if classifier.device_map is None or classifier.device_map == "single":
        classifier.to(torch.device("cuda"))

    print(f"[sanity] Loading model...")
    classifier.initialize(init_input=None)
    print_per_gpu_memory("Loaded.")

    batch = make_dummy_batch(args.batch_size, args.image_size)
    tokenized = classifier.tokenize_batch(
        images=batch["image"],
        prompts=batch["prompt"],
        responses=batch["response"],
        targets=batch["target"],
    )
    print(
        f"[sanity] tokenize_batch keys: {sorted(tokenized.keys())}; "
        f"input_ids shape={tuple(tokenized['input_ids'].shape)}"
    )

    # Move inputs to the same device as the first model parameter (HF auto
    # routes through the device map from there).
    input_device = next(classifier._model.parameters()).device
    forward_kwargs = {
        k: v.to(input_device) for k, v in tokenized.items()
        if k != "target" and isinstance(v, torch.Tensor)
    }

    for i in range(n_gpus):
        torch.cuda.reset_peak_memory_stats(i)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = classifier(**forward_kwargs)
        loss = outputs.loss
    print(f"[sanity] forward done. loss={loss.item():.4f}")
    loss.backward()
    print("[sanity] backward done.")

    nonzero_grad = 0
    total_trainable = 0
    for name, p in classifier._model.named_parameters():
        if not p.requires_grad:
            continue
        total_trainable += 1
        if p.grad is not None and torch.any(p.grad != 0):
            nonzero_grad += 1
    print(f"[sanity] trainable params with non-zero grad: {nonzero_grad}/{total_trainable}")

    print_per_gpu_memory("PEAK")

    ok = (nonzero_grad > 0) and torch.isfinite(loss).item()
    print(f"[sanity] result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
