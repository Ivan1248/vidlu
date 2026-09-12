"""Tests for separating reasoning blocks from VLM responses.

The layout that matters in practice is *closing-delimiter only*: with
``add_generation_prompt=True`` the chat template emits the opening ``<think>``
into the prompt, so the model generates only the closing tag.
"""

from types import SimpleNamespace

from vidlu_irap_gaim.vlm.models.base import BaseVLMPredictor
from vidlu_irap_gaim.vlm.models.thinking import DEFAULT_THINKING_BUDGET, split_thinking


def test_closing_delimiter_only():
    """The real Qwen3.5 layout: opening tag came from the prompt."""
    response, thinking = split_thinking("weighing it up\n</think>\n\n1: Urban\n2: None")
    assert response == "1: Urban\n2: None"
    assert thinking == "weighing it up\n</think>"


def test_both_delimiters_present():
    """Models that emit the full block still work."""
    response, thinking = split_thinking("<think>hmm</think>1: Urban")
    assert response == "1: Urban"
    assert thinking == "<think>hmm</think>"


def test_no_delimiter_returns_response_unchanged():
    """Thinking disabled: nothing to separate."""
    assert split_thinking("1: Urban\n2: None") == ("1: Urban\n2: None", None)


def test_truncated_mid_reasoning_keeps_the_reasoning():
    """No closing tag means generation was cut off inside the reasoning.

    The reasoning is returned as the response rather than being dropped, so the
    truncation is visible instead of surfacing as an empty response.
    """
    response, thinking = split_thinking("still thinking about the roadside and")
    assert response == "still thinking about the roadside and"
    assert thinking is None


def test_gemma_delimiter():
    response, thinking = split_thinking("<|channel>thought\nreasoning\n<channel|>1: Urban")
    assert response == "1: Urban"
    assert thinking == "<|channel>thought\nreasoning\n<channel|>"


def test_first_delimiter_wins():
    """Only the first closing delimiter splits; later ones stay in the response."""
    response, _ = split_thinking("a</think>b</think>c")
    assert response == "b</think>c"


def test_thinking_budget_is_shared():
    """One value for the CLI and the fine-tuning classifiers.

    The budget belongs to the model, not the eval step: the model is what splits
    generation into a reasoning phase and a response phase, so it is what needs to
    know how much reasoning to allow.
    """
    from vidlu_irap_gaim.tools import vlm_inference
    from vidlu_irap_gaim.vlm.finetuning.model import Qwen3VLClassifier

    assert vlm_inference.DEFAULT_THINKING_BUDGET is DEFAULT_THINKING_BUDGET
    assert Qwen3VLClassifier().thinking_budget == DEFAULT_THINKING_BUDGET


class _StubPredictor(BaseVLMPredictor):
    """The smallest `BaseVLMPredictor` that can report a budget."""

    def _load_model(self):
        pass

    def _generate_batch(self, pil_images, prompt, max_response_tokens):
        raise NotImplementedError

    @property
    def tokenizer(self):
        return "tokenizer"  # only passed through to the response scheme


def test_reasoning_room_is_added_to_the_response_budget():
    """A single-call backend caps reasoning and response together, so the response budget
    alone would leave reasoning nothing and the response would never be reached."""
    thinking = _StubPredictor("m", enable_thinking=True, thinking_budget=100)
    assert thinking.single_call_budget(50) == 150

    not_thinking = _StubPredictor("m", enable_thinking=False, thinking_budget=100)
    assert not_thinking.single_call_budget(50) == 50


def test_reasoning_room_applies_to_a_derived_budget_too():
    """The defect this replaced: the allowance was added to `max_response_tokens` before
    construction, so a `None` (derived per session) budget never got any."""
    scheme = SimpleNamespace(
        compute_max_response_tokens=lambda tokenizer, attrs, margin_tokens: 40)
    predictor = _StubPredictor("m", max_response_tokens=None, enable_thinking=True,
                               thinking_budget=100)

    response_budget = predictor.session_response_budget(scheme, ["a", "b"])

    # The response budget itself stays the response's – `VLMClassifierPredictor` hands it to a
    # classifier that budgets the two phases separately.
    assert response_budget == 40
    assert predictor.single_call_budget(response_budget) == 140


def test_reasoning_that_never_reached_a_response_is_reported_as_truncated():
    """Otherwise the reasoning is parsed as the response and scored as a wrong one."""
    predictor = _StubPredictor("m", enable_thinking=True)

    response, thinking, is_truncated = predictor.split_thinking_and_truncation(
        "still weighing up the roadside and", is_truncated=False)
    assert (response, thinking, is_truncated) == ("still weighing up the roadside and", None, True)

    response, thinking, is_truncated = predictor.split_thinking_and_truncation(
        "weighing it up\n</think>\n\n1: Urban", is_truncated=False)
    assert (response, is_truncated) == ("1: Urban", False)
    assert thinking is not None


def test_an_unreported_truncation_stays_unknown_when_the_response_arrived():
    """vLLM reports why it stopped; a backend that does not says None, which is not False."""
    predictor = _StubPredictor("m", enable_thinking=True)
    assert predictor.split_thinking_and_truncation(
        "reasoning</think>1: Urban", is_truncated=None)[2] is None


def test_without_thinking_the_response_and_the_truncation_flag_pass_through():
    predictor = _StubPredictor("m", enable_thinking=False)
    assert predictor.split_thinking_and_truncation(
        "1: Urban", is_truncated=False) == ("1: Urban", None, False)
