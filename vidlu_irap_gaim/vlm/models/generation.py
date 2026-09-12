"""
Shared generation bookkeeping for the two VLM hierarchies.

`_BaseVLMClassifier` (a trainable model) and `BaseHFPredictor` (a zero-shot
predictor) both call HF ``generate`` and both have to decide what stopping
means, so that decision lives here rather than in two copies that can drift.
"""

from contextlib import nullcontext
import typing as T
import warnings

import torch


def bf16_autocast(enabled: bool):
    """The mixed-precision context generation and the forward passes run under."""
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def get_text_tokenizer(processor):
    """The text tokenizer of a processor.

    Some processors *are* the tokenizer rather than wrapping one, so the two
    hierarchies would otherwise each carry their own copy of this rule.
    """
    return getattr(processor, "tokenizer", processor)


def get_eos_token_ids(model, tokenizer) -> set[int]:
    """Token ids that stop generation.

    ``generation_config`` is authoritative and may list several – Qwen stops on
    ``<|im_end|>``, not only on the tokenizer's nominal EOS – so reading the
    tokenizer alone would under-report and make complete responses look
    truncated.
    """
    ids: set[int] = set()
    for source in (getattr(model, "generation_config", None), tokenizer):
        eos = getattr(source, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif eos is not None:
            ids.update(int(i) for i in eos)
    return ids


def find_truncated_rows(generated_ids: torch.Tensor, budget: int, eos_ids: set[int]) -> list[bool]:
    """Per row: whether generation was cut off by the cap rather than stopping itself.

    Length alone over-reports: a response whose final allowed token *is* a stop
    token used its whole budget yet is complete. In a batch the final position is
    not informative either, because ``generate`` runs until every row is done and
    pads out the rows that finished earlier. A row is complete exactly when it
    contains a stop token anywhere.
    """
    if generated_ids.shape[1] < budget:
        return [False] * generated_ids.shape[0]
    eos = torch.tensor(sorted(eos_ids), device=generated_ids.device)
    return (~torch.isin(generated_ids, eos).any(dim=1)).tolist()


def warn_if_truncated(is_truncated_per_row: T.Sequence[bool | None], budget: int | None) -> None:
    """Warns once per batch when some rows hit the response budget without stopping.

    A derived budget (``budget=None`` at the caller, resolved from the response
    scheme) is a bound no compliant response can exceed, so hitting it means the
    model is emitting something the format does not allow – commentary,
    repetition – and the responses are what to inspect, not the budget.
    """
    num_truncated = sum(1 for t in is_truncated_per_row if t)
    if num_truncated:
        warnings.warn(
            f"{num_truncated}/{len(is_truncated_per_row)} responses hit the {budget}-token"
            f" response budget without reaching EOS, so they are truncated."
            f" When the budget is derived from the"
            f" response scheme, a compliant response cannot do this – so the model is emitting"
            f" something the format does not allow. Inspect the responses rather than simply"
            f" raising the budget.")
