"""Tests for VLMEvalStep's generative-eval bookkeeping.

Covers the contract that makes a failed generative eval diagnosable: because a
missing prediction is silently scored as class 0, `out` alone cannot distinguish
a genuine class-0 prediction from a failure, so the step also reports the raw
decoded text and the per-sample error.

Pure unit tests: a stub model and stub response scheme, no weights or GPU.
"""

import pytest
import torch
from torch import nn

from vidlu_irap_gaim.vlm.response_parser import AttributePrediction
from vidlu_irap_gaim.vlm.models.thinking import DEFAULT_THINKING_BUDGET, split_thinking
from vidlu_irap_gaim.vlm.finetuning.steps import _generate_and_parse_batch

GENERATION_FAILS = "<generation-fails>"
PARSING_FAILS = "<parsing-fails>"


class _WordTokenizer:
    """Whitespace tokenizer; one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


class _StubModel(nn.Module):
    """Returns the prompt as the response, so tests choose it per sample.

    Mirrors the real contract: `generate_batch_for_eval` generates for a whole batch
    with one (response, thinking, is_truncated) per input, the reasoning already
    separated out, and `generate_for_eval` is its one-element case.
    """

    def __init__(self, enable_thinking=None, is_truncated=False,
                 thinking_budget=DEFAULT_THINKING_BUDGET):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.enable_thinking = enable_thinking
        # The classifier owns the reasoning budget, as `_BaseVLMClassifier` does; a
        # predictor wrapping it reads this rather than keeping its own copy.
        self.thinking_budget = thinking_budget
        self.is_truncated = is_truncated
        self.tokenizer = _WordTokenizer()
        self.last_prompts = None

    def generate_batch_for_eval(self, images, prompts, max_response_tokens=512, amp=True,
                                min_new_tokens=0):
        prompts = list(prompts)
        if any(prompt == GENERATION_FAILS for prompt in prompts):
            raise RuntimeError("CUDA hiccup")
        self.last_prompts = prompts
        self.last_max_response_tokens = max_response_tokens
        self.last_min_new_tokens = min_new_tokens
        return [(*(split_thinking(prompt) if self.enable_thinking else (prompt, None)),
                 self.is_truncated)
                for prompt in prompts]

    def generate_for_eval(self, image, prompt, max_response_tokens=512, amp=True,
                          min_new_tokens=0):
        return self.generate_batch_for_eval(
            [image], [prompt], max_response_tokens=max_response_tokens, amp=amp,
            min_new_tokens=min_new_tokens)[0]


class _StubScheme:
    attr_to_value_to_class_idx = {"A": {"x": 0, "y": 1}, "B": {"p": 0, "q": 1}}

    def __init__(self):
        self.parsed_texts = []

    def parse_response(self, response_text, attrs_to_include):
        self.parsed_texts.append(response_text)
        if response_text == PARSING_FAILS:
            raise ValueError("unparseable response")
        return {"A": AttributePrediction(attr_name="A", pred_value="y", pred_idx=1)}


def _run(prompts, enable_thinking=None, scheme=None, is_truncated=False):
    return _generate_and_parse_batch(
        model=_StubModel(enable_thinking=enable_thinking, is_truncated=is_truncated),
        images=torch.zeros(len(prompts), 3, 4, 4),
        prompts=prompts,
        targets=torch.zeros(len(prompts), 2, dtype=torch.long),
        response_scheme=scheme if scheme is not None else _StubScheme(),
        attrs_to_include=["A", "B"],
    )


def test_responses_are_returned_without_wrapping_generate():
    result = _run(["hello", "world"])
    assert result.responses == ["hello", "world"]
    assert result.thinking_texts == [None, None]
    assert result.is_truncated_per_row == [False, False]
    assert result.response_errors == [None, None]


# ----- truncation ---------------------------------------------------------------

def test_truncated_flag_is_reported():
    """`generate_for_eval` decides truncation from the raw token count; the step
    just has to pass it through."""
    assert _run(["hello"], is_truncated=False).is_truncated_per_row == [False]
    assert _run(["hello"], is_truncated=True).is_truncated_per_row == [True]


def test_generation_failure_leaves_truncated_unknown():
    """No budget was consumed to observe when generation itself raised."""
    result = _run([GENERATION_FAILS])
    assert result.is_truncated_per_row == [None]


def test_truncated_response_warns():
    with pytest.warns(UserWarning, match="truncated"):
        _run(["hello"], is_truncated=True)


# ----- thinking ---------------------------------------------------------------

def test_thinking_is_separated_from_the_response():
    result = _run(["weighing it up\n</think>\n\n1: Urban"], enable_thinking=True)
    assert result.responses == ["1: Urban"]
    assert result.thinking_texts == ["weighing it up\n</think>"]


def test_parsing_never_sees_the_reasoning():
    """The `</think>` prefix would otherwise make parse_response fail, and the
    failure would be swallowed into class-0 predictions."""
    scheme = _StubScheme()
    _run(["reasoning\n</think>\n\n1: Urban"], enable_thinking=True, scheme=scheme)
    assert scheme.parsed_texts == ["1: Urban"]
    assert all("</think>" not in t for t in scheme.parsed_texts)


def test_generation_failure_has_no_text():
    """No text exists when generation itself raised."""
    result = _run([GENERATION_FAILS])
    assert result.responses == [None]
    assert "CUDA hiccup" in result.response_errors[0]


def test_parse_failure_keeps_the_text():
    """The text survives a parser failure – this is what separates a truncated
    response from a parser bug, which `out` alone cannot show."""
    result = _run([PARSING_FAILS])
    assert result.responses == [PARSING_FAILS]
    assert "unparseable response" in result.response_errors[0]


def test_failed_sample_is_indistinguishable_in_out_alone():
    """Pins the reason the extra fields exist: a failure scores as class 0,
    exactly like a real class-0 prediction."""
    ok, failed = _run(["fine"]), _run([PARSING_FAILS])
    attr_b = list(_StubScheme.attr_to_value_to_class_idx).index("B")
    # "B" is never predicted by the stub scheme, so it falls back to class 0 --
    # the same value a failed sample gets for every attribute.
    assert ok.out[attr_b].argmax(1).tolist() == failed.out[attr_b].argmax(1).tolist() == [0]
    # The distinction is only visible in response_errors.
    assert ok.response_errors == [None] and failed.response_errors != [None]


# ----- response budget ------------------------------------------------------------

class _BudgetScheme(_StubScheme):
    def compute_max_response_tokens(self, tokenizer, attrs_to_include):
        self.budget_args = (tokenizer, list(attrs_to_include))
        return 123


def _step_with(scheme, attrs=("A", "B"), **kwargs):
    from vidlu_irap_gaim.vlm.finetuning.dataset import VLMDatasetConfig
    from vidlu_irap_gaim.vlm.finetuning.steps import VLMEvalStep

    step = VLMEvalStep(**kwargs)
    # Normally read off the dataset's `info` by `_ensure_config`.
    step._config = VLMDatasetConfig(
        response_scheme=scheme,
        attrs_to_include=list(attrs),
        detail_level="attr_desc_vals",
        attr_to_value_to_class_idx=scheme.attr_to_value_to_class_idx)
    return step


def test_response_budget_is_derived_from_the_scheme():
    """The scheme knows the format and the closed value sets, so it owns the bound
    – the step must not carry a hardcoded guess."""
    scheme = _BudgetScheme()
    step = _step_with(scheme)
    assert step.max_response_tokens is None
    assert step.get_response_budget(_StubModel()) == 123
    assert scheme.budget_args[1] == ["A", "B"]


def test_derived_budget_ignores_thinking():
    """Reasoning gets its own budget inside the model, so the response bound must
    not change when thinking is on."""
    step = _step_with(_BudgetScheme())
    assert step.get_response_budget(_StubModel(enable_thinking=True)) == 123
    assert step.get_response_budget(_StubModel(enable_thinking=False)) == 123


def test_explicit_budget_overrides_the_derived_one():
    scheme = _BudgetScheme()
    step = _step_with(scheme, max_response_tokens=77)
    assert step.get_response_budget(_StubModel()) == 77
    assert not hasattr(scheme, "budget_args"), "scheme should not be consulted"


def test_out_and_target_still_metric_shaped():
    """Adding fields must not disturb what the metrics read."""
    result = _run(["a", "b"])
    assert len(result.out) == len(_StubScheme.attr_to_value_to_class_idx)
    assert result.out[0].shape == (2, 2)
    assert result.target.shape == (2, 2)


def test_the_predictor_delegates_generation_to_the_wrapped_classifier():
    """`VLMClassifierPredictor` used to hand-copy `generate_for_eval`, hardcoding Qwen's
    vision preprocessing; it must go through the classifier's own path, min_new_tokens
    included."""
    from PIL import Image

    from vidlu_irap_gaim.vlm.finetuning.predictor import VLMClassifierPredictor

    model = _StubModel()
    model.model_id = "stub/model"
    predictor = VLMClassifierPredictor(model, max_response_tokens=64, min_new_tokens=7)

    (response, thinking, is_truncated), = predictor._generate_batch(
        [Image.new("RGB", (4, 4))], "hello world", 64)

    assert (response, thinking, is_truncated) == ("hello world", None, False)
    assert model.last_max_response_tokens == 64
    assert model.last_min_new_tokens == 7


def test_the_predictor_asks_one_prompt_of_every_image_in_the_batch():
    """The session loop batches along the fixed-prompt axis, so the classifier must
    receive that one prompt repeated – not a per-image prompt list."""
    from PIL import Image

    from vidlu_irap_gaim.vlm.finetuning.predictor import VLMClassifierPredictor

    model = _StubModel()
    model.model_id = "stub/model"
    predictor = VLMClassifierPredictor(model, max_response_tokens=64)

    images = [Image.new("RGB", (4, 4)) for _ in range(3)]
    results = predictor._generate_batch(images, "one question", 64)

    assert len(results) == 3
    assert model.last_prompts == ["one question"] * 3
