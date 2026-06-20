"""
Post-training generative evaluation for a fine-tuned Gemma 4 VLM.

The training loop short-circuits the per-sample autoregressive eval via
``VLM_SKIP_GENERATIVE_EVAL=1`` (see ``VLMEvalStep``).  This script
performs the deferred generative scoring as a single-process pass:

  1. Build the same VLMIrapDataset that the trainer used.
  2. Load a ``Gemma4VLClassifier`` (naive MP across visible GPUs via
     ``device_map="auto"``) and apply the saved adapter weights.
  3. Walk the requested split, call ``model.generate_for_eval`` per sample,
     parse responses through the dataset's ``ResponseScheme``, and update
     the standard IRAP attribute metrics.

Usage (4× A6000 node):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_generative_gemma.py \\
        --checkpoint /path/to/adapter_state.pt \\
        --split val

The checkpoint file is the state_dict produced by Vidlu's
``CheckpointManager`` (or any pickle of the dict returned by
``Gemma4VLClassifier.state_dict``).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from vidlu_irap_gaim.vlm.finetuning import (
    Gemma4VLClassifier,
    make_vlm_bih_data,
)
from vidlu_irap_gaim.vlm.predictions import convert_attribute_predictions_to_standard_format
from irap_data.attrs import get_attrs_to_include
from vidlu_irap_gaim.metrics import get_irap_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to adapter checkpoint (state_dict pickle).")
    p.add_argument("--model-id", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--split", default="val",
                   help="Dataset split name to evaluate (default: %(default)s).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on the number of samples (debug).")
    p.add_argument("--max-response-tokens", type=int, default=256)
    p.add_argument("--response-scheme", default=None,
                   help="Optional override; defaults to make_vlm_bih_data's default.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[eval] CUDA is not available; aborting.", file=sys.stderr)
        return 2

    # 1. Dataset
    data_kwargs = {}
    if args.response_scheme:
        data_kwargs["response_scheme_name"] = args.response_scheme
    data = make_vlm_bih_data(**data_kwargs)
    if args.split not in data:
        print(f"[eval] split {args.split!r} not in dataset; have {list(data.keys())}",
              file=sys.stderr)
        return 3
    dataset = data[args.split]
    response_scheme = dataset.info.vlm_response_scheme
    attrs_to_include = list(get_attrs_to_include())

    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"[eval] split={args.split!r} samples={n} model_id={args.model_id}")

    # 2. Model + adapter
    classifier = Gemma4VLClassifier(model_id=args.model_id, lora_r=args.lora_r)
    classifier.initialize(init_input=None)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classifier.load_state_dict(state)
    classifier.eval()

    # 3. Metrics
    metrics = get_irap_metrics()
    for m in metrics:
        m.reset()

    # 4. Walk the split
    import warnings
    for i in range(n):
        sample = dataset[i]
        image = sample["image"]
        prompt = sample["prompt"]
        target = sample["target"]
        try:
            response_text = classifier.generate_for_eval(
                image=image,
                prompt=prompt,
                max_response_tokens=args.max_response_tokens,
                amp=True,
            )
            predictions = response_scheme.parse_response(response_text, attrs_to_include)
        except Exception as e:
            warnings.warn(f"Generation/parsing failed for sample {i}: {e}")
            predictions = {}

        out_list = convert_attribute_predictions_to_standard_format(
            [predictions],
            response_scheme.attr_to_value_to_class_idx,
            list(response_scheme.attr_to_value_to_class_idx.keys()),
            device=torch.device("cuda"),
            attrs_to_include=attrs_to_include,
        )
        target_tensor = (target if isinstance(target, torch.Tensor) else torch.as_tensor(target))
        target_tensor = target_tensor.unsqueeze(0).to("cuda")

        from vidlu.utils.collections import NameDict
        for m in metrics:
            m.update(NameDict(out=out_list, target=target_tensor))

        if (i + 1) % 25 == 0:
            print(f"[eval] {i + 1}/{n} done")

    # 5. Report
    print("\n[eval] ==== metrics ====")
    for m in metrics:
        value = m.compute()
        if isinstance(value, dict):
            for k, v in value.items():
                v_str = f"{float(v):.4f}" if hasattr(v, "__float__") else str(v)
                print(f"  {k}: {v_str}")
        else:
            print(f"  {m.name}: {float(value):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
