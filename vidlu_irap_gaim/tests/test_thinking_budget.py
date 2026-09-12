"""Tests for `_BaseVLMClassifier._generate_within_budget`'s reasoning/response split.

Reasoning is generated before the response and shares the same `generate()` cap, so
a single call can spend everything reasoning and return no response – and having
emitted no closing delimiter, that reasoning is then indistinguishable from an
response and gets parsed as one.  `_generate_within_budget` prevents that by budgeting
reasoning separately and forcing the block closed when the reasoning budget runs
out, so the response always gets its own full allowance.

Pure unit tests: a stub HF-like model that emits scripted token sequences, plus a
stub processor.  No weights, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vidlu_irap_gaim.vlm.finetuning.model import _BaseVLMClassifier
from vidlu_irap_gaim.vlm.models.thinking import QWEN_THINKING_END

# One "token" per vocabulary entry; ids index into this list so decoding is
# just a lookup and scripted generations stay readable.
VOCAB = ["<pad>", "reasoning", QWEN_THINKING_END, "1:", "Urban", "<eos>"]
ID = {tok: i for i, tok in enumerate(VOCAB)}

# Small enough that the budget an unlimited request resolves to is hand-checkable.
CONTEXT_LENGTH = 10


class _StubTokenizer:
    eos_token_id = ID["<eos>"]

    def encode(self, text, add_special_tokens=False):
        return [ID[text]]


class _StubProcessor:
    tokenizer = _StubTokenizer()

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(VOCAB[int(i)] for i in row) for row in ids]


class _Config:
    """As much of an HF config as the budget resolution reads."""

    def __init__(self, context_length):
        self.text_config = SimpleNamespace(max_position_embeddings=context_length)


class _ScriptedModel:
    """Emits a preset continuation per `generate` call, capped at max_new_tokens.

    Each script entry is the full continuation the model "wants" to emit; ending
    it with `<eos>` stands for stopping voluntarily, and being cut by the cap
    stands for truncation.
    """

    generation_config = None

    def __init__(self, scripts, context_length=CONTEXT_LENGTH):
        self.scripts = list(scripts)
        self.calls = []
        self.config = _Config(context_length)

    def generate(self, input_ids=None, max_new_tokens=None, **kwargs):
        script = self.scripts.pop(0)
        emitted = script[:max_new_tokens]
        self.calls.append({"prompt_len": input_ids.shape[1],
                           "max_new_tokens": max_new_tokens,
                           "prompt_ids": input_ids[0].tolist()})
        new = torch.tensor([[ID[t] for t in emitted]], dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


class _Classifier(_BaseVLMClassifier):
    """Exposes `_generate_within_budget` over the stubs."""

    def __init__(self, scripts, *, enable_thinking, thinking_budget):
        super().__init__(enable_thinking=enable_thinking, thinking_budget=thinking_budget)
        self._model = _ScriptedModel(scripts)
        self._processor = _StubProcessor()


def _inputs(prompt_len=2):
    ids = torch.zeros(1, prompt_len, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _run(scripts, *, enable_thinking, thinking_budget=3, response_budget=3):
    model = _Classifier(scripts, enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget)
    # `_generate_within_budget` generates for a whole batch; these tests drive one row.
    (text,), (is_truncated,) = model._generate_within_budget(_inputs(), response_budget, amp=False)
    return text, is_truncated, model._model.calls


# ----- thinking disabled ------------------------------------------------------

def test_without_thinking_a_single_call_is_made():
    text, is_truncated, calls = _run([["1:", "Urban", "<eos>"]], enable_thinking=False)
    assert text == "1: Urban <eos>"
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 3
    assert is_truncated is False


def test_without_thinking_hitting_the_cap_is_truncation():
    text, is_truncated, calls = _run(
        [["1:", "Urban", "1:", "Urban"]], enable_thinking=False, response_budget=3
    )
    assert is_truncated is True
    assert len(calls) == 1


def test_spending_the_whole_budget_but_stopping_is_not_truncation():
    """Length alone over-reports: a response whose last allowed token is the stop
    token used its entire budget yet is complete."""
    complete = ["1:", "Urban", "<eos>"]
    cut_off = ["1:", "Urban", "Urban"]
    assert _run([complete], enable_thinking=False, response_budget=3)[1] is False
    assert _run([cut_off], enable_thinking=False, response_budget=3)[1] is True


# ----- thinking, reasoning finishes in budget ---------------------------------

def test_reasoning_that_closes_and_finishes_needs_one_call():
    """Stopping under the reasoning cap means EOS, so nothing more is needed."""
    text, is_truncated, calls = _run(
        [["reasoning", QWEN_THINKING_END]], enable_thinking=True, thinking_budget=3
    )
    assert len(calls) == 1
    assert is_truncated is False
    assert QWEN_THINKING_END in text


# ----- thinking, reasoning exhausts its budget --------------------------------

def test_exhausted_reasoning_gets_the_delimiter_forced_in():
    """The defect this prevents: no closing delimiter, so the reasoning would be
    parsed as the response and `thinking_texts` would report None."""
    text, is_truncated, calls = _run(
        [["reasoning", "reasoning", "reasoning"], ["1:", "Urban", "<eos>"]],
        enable_thinking=True, thinking_budget=3, response_budget=3,
    )
    assert len(calls) == 2, "must continue after exhausting the reasoning budget"
    # The forced delimiter is appended to the prefix of the second call.
    assert calls[1]["prompt_ids"][-1] == ID[QWEN_THINKING_END]
    assert QWEN_THINKING_END in text
    assert text.endswith("1: Urban <eos>")
    assert is_truncated is False


def test_the_response_gets_its_full_budget_after_forced_close():
    """The guarantee: exhausting the reasoning budget costs reasoning quality
    only – the response allowance is not reduced by what reasoning consumed."""
    _, _, calls = _run(
        [["reasoning", "reasoning", "reasoning"], ["1:", "Urban", "<eos>"]],
        enable_thinking=True, thinking_budget=3, response_budget=7,
    )
    assert calls[0]["max_new_tokens"] == 3
    assert calls[1]["max_new_tokens"] == 7


def test_no_delimiter_is_forced_when_the_model_already_closed():
    """Reasoning that closed but left no room for the response still continues --
    just without a redundant delimiter."""
    _, _, calls = _run(
        [["reasoning", QWEN_THINKING_END, "1:"], ["Urban", "<eos>"]],
        enable_thinking=True, thinking_budget=3, response_budget=3,
    )
    assert len(calls) == 2
    assert calls[1]["prompt_ids"][-1] == ID["1:"], "no delimiter should be appended"


def test_truncation_after_forced_close_reflects_the_response_only():
    text, is_truncated, calls = _run(
        [["reasoning", "reasoning", "reasoning"], ["1:", "Urban", "1:", "Urban"]],
        enable_thinking=True, thinking_budget=3, response_budget=3,
    )
    assert len(calls) == 2
    assert is_truncated is True, "the response, not the reasoning, overran"


# ----- no response limit --------------------------------------------------------

def test_an_unlimited_response_gets_the_rest_of_the_context_window():
    """`response_budget=None` is not "no cap at all": `generate` needs a number, and the honest
    one is what the context window still has room for after the prompt."""
    text, is_truncated, calls = _run([["1:", "Urban", "<eos>"]], enable_thinking=False,
                                     response_budget=None)
    assert calls[0]["max_new_tokens"] == CONTEXT_LENGTH - 2  # `_inputs` uses a 2-token prompt
    assert text == "1: Urban <eos>"
    assert is_truncated is False


def test_filling_the_context_window_still_counts_as_truncation():
    text, is_truncated, calls = _run([["Urban"] * CONTEXT_LENGTH], enable_thinking=False,
                                     response_budget=None)
    assert is_truncated is True


def test_an_unlimited_response_after_reasoning_excludes_the_reasoning_from_its_budget():
    """The reasoning block shares the window, so the response's share is what is left after it."""
    scripts = [["reasoning", "reasoning", "reasoning"],  # exhausts the 3-token reasoning budget
               ["1:", "Urban", "<eos>"]]
    _, _, calls = _run(scripts, enable_thinking=True, thinking_budget=3, response_budget=None)
    # 2 prompt + 3 reasoning + 1 forced delimiter = 6 tokens of prefix.
    assert calls[1]["prompt_len"] == 6
    assert calls[1]["max_new_tokens"] == CONTEXT_LENGTH - 6


def test_a_prompt_that_fills_the_context_window_is_reported():
    model = _Classifier([["<eos>"]], enable_thinking=False, thinking_budget=3)
    with pytest.raises(ValueError, match="context window"):
        model._generate_within_budget(_inputs(prompt_len=CONTEXT_LENGTH), None, amp=False)


def test_the_context_length_can_come_from_a_flat_config():
    """Text-only configs keep `max_position_embeddings` at the top level."""
    model = _Classifier([["<eos>"]], enable_thinking=False, thinking_budget=3)
    model._model.config = SimpleNamespace(max_position_embeddings=64)
    assert model._resolve_response_budget(None, 4) == 60


def test_a_config_without_a_declared_context_length_is_reported():
    model = _Classifier([["<eos>"]], enable_thinking=False, thinking_budget=3)
    model._model.config = SimpleNamespace()
    with pytest.raises(RuntimeError, match="max_position_embeddings"):
        model._resolve_response_budget(None, 4)
