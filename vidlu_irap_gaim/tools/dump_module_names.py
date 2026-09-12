"""Inspects linear submodules in a base VLM to identify candidate LoRA target module names."""

import argparse
from collections import defaultdict
from typing import Any

from torch import nn


def dump_module_names(classifier_class, model_id: str | None = None, *,
                      load_in_4bit: bool = False, max_examples: int = 3) -> dict[str, Any]:
    """Inspects linear submodules in a base VLM to identify candidate LoRA target module names.

    Args:
        classifier_class: Subclass of `_BaseVLMClassifier` (e.g. `Qwen3VLClassifier`).
        model_id: Hugging Face model repository identifier (defaults to class default).
        load_in_4bit: Whether to quantize weights in 4-bit mode.
        max_examples: Maximum number of example module path names to store per suffix.

    Returns:
        Dictionary containing `counts`, `types`, and `examples` mapping per module suffix.
    """
    inst = classifier_class(model_id=model_id, load_in_4bit=load_in_4bit)
    # 4-bit must materialize on CUDA; bf16 name-inspection stays on CPU so it needs no GPU and
    # does not disturb a training process.
    load_kwargs = inst.make_load_kwargs(device_map={"": "cuda"} if load_in_4bit else {"": "cpu"})
    # Name inspection never runs a forward pass, so the cheapest attention impl will do.
    load_kwargs["attn_implementation"] = "eager"

    print(f"[dump_module_names] loading {inst.model_id} (load_in_4bit={load_in_4bit})...")
    model = inst._build_hf_model(load_kwargs)

    counts: dict[str, int] = defaultdict(int)
    types: dict[str, set] = defaultdict(set)
    examples: dict[str, list] = defaultdict(list)
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or "Linear" in type(module).__name__):
            continue
        suffix = name.rsplit(".", 1)[-1]
        counts[suffix] += 1
        types[suffix].add(type(module).__name__)
        if len(examples[suffix]) < max_examples:
            examples[suffix].append(name)

    print(f"[dump_module_names] {inst.model_id}: {len(counts)} linear-like suffixes")
    for suffix in sorted(counts, key=lambda s: -counts[s]):
        type_str = ",".join(sorted(types[suffix]))
        example_str = " | ".join(examples[suffix])
        print(f"  {suffix:24s} x{counts[suffix]:<5d} [{type_str}]  e.g. {example_str}")
    return {
        "counts": dict(counts),
        "types": {k: sorted(v) for k, v in types.items()},
        "examples": dict(examples),
    }


def main():
    from vidlu_irap_gaim.vlm import finetuning

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("classifier_class",
                        help="Name of a classifier in vidlu_irap_gaim.vlm.finetuning,"
                             " e.g. Qwen3VLClassifier")
    parser.add_argument("--model-id", default=None, help="HF model id (default: class default)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize while loading (requires CUDA)")
    args = parser.parse_args()

    dump_module_names(getattr(finetuning, args.classifier_class), args.model_id,
                      load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    main()
