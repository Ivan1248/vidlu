"""Tests for VLM fine-tuning tokenization: label masking and chat-template handling.

All tests here are pure unit tests driven by stub processors/tokenizers – no
weights, no GPU, no network. The stubs reproduce the *real* Qwen chat-template
branches (quoted in the docstrings below) so that a template regression shows up
as a test failure rather than as silently mis-masked labels.
"""

import pytest
import torch

from vidlu_irap_gaim.vlm.finetuning.model import (
    Qwen35Classifier,
    Qwen3VLClassifier,
    _BaseVLMClassifier,
    _compute_response_start_positions,
    _create_labels_with_prompt_mask,
)


# ----- stub tokenizer / processor ---------------------------------------------

class _WordTokenizer:
    """Whitespace tokenizer; one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


class _StubProcessor:
    """Minimal stand-in for Qwen3VLProcessor's ``apply_chat_template``.

    Mirrors the two branches of the real Qwen3.5 template that decide where the
    response starts::

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\\n\\n</think>\\n\\n' }}
            {%- else %}
                {{- '<think>\\n' }}
            {%- endif %}
        {%- endif %}

        {%- elif message.role == "assistant" %}
            {%- if loop.index0 > ns.last_query_index %}
                {{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content
                    + '\\n</think>\\n\\n' + content }}

    Tokens are space-separated so ``_WordTokenizer`` can count them.
    """

    def __init__(self, thinking_capable=True):
        self.thinking_capable = thinking_capable
        self.tokenizer = _WordTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None
    ):
        prompt = messages[0]["content"][-1]["text"]
        text = f"<|im_start|> user {prompt} <|im_end|> <|im_start|> assistant"
        if self.thinking_capable:
            # The closed block is emitted both by the assistant-message branch
            # (always) and by the generation prompt when thinking is disabled.
            if add_generation_prompt and enable_thinking is not False:
                return f"{text} <think>"
            text = f"{text} <think> </think>"
        if add_generation_prompt:
            return text
        return f"{text} {messages[1]['content']} <|im_end|>"


class _StubClassifier(_BaseVLMClassifier):
    """Exposes the shared render/assemble helpers with a stub processor."""

    def __init__(self, processor, enable_thinking=None):
        super().__init__(enable_thinking=enable_thinking)
        self._processor = processor


def _messages(prompt, response=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


# ----- response boundary ------------------------------------------------------

def test_response_start_right_padded():
    # 2 samples, 6 slots. Sample 0 has 5 real tokens, sample 1 has 4.
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    positions = _compute_response_start_positions(
        attention_mask, response_texts=["r1 r2", "r1"], tokenizer=_WordTokenizer()
    )
    assert positions == [3, 3]


def test_response_start_left_padded():
    # Same sequences, padded on the left instead.
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    positions = _compute_response_start_positions(
        attention_mask, response_texts=["r1 r2", "r1"], tokenizer=_WordTokenizer()
    )
    # first real token (1 / 2) + prompt length (3 / 3)
    assert positions == [4, 5]


def test_labels_mask_prompt_and_padding():
    input_ids = torch.arange(12).reshape(2, 6)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    labels = _create_labels_with_prompt_mask(input_ids, attention_mask, [3, 3])
    # Only the response slots survive.
    assert labels[0].tolist() == [-100, -100, -100, 3, 4, -100]
    assert labels[1].tolist() == [-100, -100, -100, 9, -100, -100]


# ----- thinking mode ----------------------------------------------------------

def test_thinking_disabled_gives_clean_response():
    """With enable_thinking=False the supervised span is exactly the response."""
    model = _StubClassifier(_StubProcessor(), enable_thinking=False)
    texts_full, texts_prompt_only = model._render_chat_texts(
        [_messages("p", "the response")], _messages("p")
    )
    response = texts_full[0][len(texts_prompt_only[0]):]
    assert response == " the response <|im_end|>"
    assert "</think>" not in response


def test_thinking_enabled_leaks_close_tag_into_response():
    """Left enabled, the supervised span carries `</think>` scaffolding.

    This is the defect `enable_thinking=False` prevents: the generation prompt
    stops inside an open think block, so the model is trained to emit the
    closing tag and eval output starts with it.
    """
    model = _StubClassifier(_StubProcessor(), enable_thinking=True)
    texts_full, texts_prompt_only = model._render_chat_texts(
        [_messages("p", "the response")], _messages("p")
    )
    response = texts_full[0][len(texts_prompt_only[0]):]
    assert "</think>" in response


def test_enable_thinking_none_omits_the_kwarg():
    """None must leave the checkpoint's own default in force."""
    seen = {}

    class _Recording(_StubProcessor):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, **kwargs):
            seen.update(kwargs)
            return super().apply_chat_template(
                messages, tokenize, add_generation_prompt, **kwargs)

    _StubClassifier(_Recording(), enable_thinking=None)._render_chat_texts(
        [_messages("p", "the response")], _messages("p"))
    assert "enable_thinking" not in seen

    _StubClassifier(_Recording(), enable_thinking=False)._render_chat_texts(
        [_messages("p", "the response")], _messages("p"))
    assert seen["enable_thinking"] is False


def test_non_thinking_template_unaffected_by_kwarg():
    """Qwen3-VL-Instruct's template has no think branch, so the kwarg is a no-op."""
    plain = _StubProcessor(thinking_capable=False)
    without = _StubClassifier(plain)._render_chat_texts(
        [_messages("p", "the response")], _messages("p")
    )
    with_kwarg = _StubClassifier(plain, enable_thinking=False)._render_chat_texts(
        [_messages("p", "the response")], _messages("p")
    )
    assert without == with_kwarg
    assert without[0][0][len(without[1][0]):] == " the response <|im_end|>"


# ----- prefix guard -----------------------------------------------------------

class _DivergingProcessor(_StubProcessor):
    """Generation prompt is not a prefix of the full render."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None):
        if add_generation_prompt:
            return "<|im_start|> assistant <think>"
        return "<|im_start|> assistant nope"


def test_prefix_violation_raises():
    model = _StubClassifier(_DivergingProcessor())
    with pytest.raises(RuntimeError, match="prompt-prefix assumption"):
        model._render_chat_texts([_messages("p", "r")], _messages("p"))


# ----- Qwen3.5 configuration --------------------------------------------------

def test_qwen35_reuses_qwen3vl_path():
    """Qwen3.5 shares Qwen3VLProcessor, so it must not fork the tokenize path."""
    assert issubclass(Qwen35Classifier, Qwen3VLClassifier)
    for method in ("tokenize_batch", "_build_generation_inputs", "generate_for_eval",
                   "build_messages"):
        assert method not in vars(Qwen35Classifier), f"{method} should be inherited"


def test_qwen35_opts_out_of_left_padded_generation():
    """Its Gated-DeltaNet recurrence is only correct with right padding, the same
    constraint `tokenize_batch` pins `padding_side="right"` for.  Generation needs
    left padding, so a ragged batch must fall back to row-by-row."""
    assert Qwen35Classifier._SUPPORTS_LEFT_PADDED_GENERATION is False
    assert Qwen3VLClassifier._SUPPORTS_LEFT_PADDED_GENERATION is True


def test_qwen35_defaults():
    assert Qwen35Classifier._DEFAULT_MODEL_ID == "Qwen/Qwen3.5-9B"
    # head_dim=256 trips many flash-attn builds at forward time.
    assert "flash_attention_2" not in Qwen35Classifier._ATTN_IMPL_PREFS
    # Dense checkpoint: no MoE experts for bitsandbytes to skip.
    assert Qwen35Classifier._DEFAULT_LOAD_IN_4BIT is True
    # Qwen3.5 reasons by default; training targets contain no reasoning.
    assert Qwen35Classifier._DEFAULT_ENABLE_THINKING is False
    assert Qwen35Classifier(enable_thinking=True).enable_thinking is True


def test_qwen35_lora_targets_cover_all_three_module_families():
    targets = set(Qwen35Classifier._DEFAULT_LORA_TARGET_MODULES)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets  # Gated-Attention
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"} <= targets  # DeltaNet
    assert "out_proj" in targets  # PEFT's `.o_proj` suffix match misses this
    assert {"gate_proj", "up_proj", "down_proj"} <= targets  # dense FFN


def test_non_reasoning_classifiers_omit_the_kwarg():
    """Qwen3-VL and Gemma 4 must keep their previous byte-identical behaviour:
    no `enable_thinking` passed at all, so the checkpoint default stands."""
    assert _BaseVLMClassifier._DEFAULT_ENABLE_THINKING is None
    assert Qwen3VLClassifier._DEFAULT_ENABLE_THINKING is None
    assert Qwen3VLClassifier()._chat_template_kwargs() == {}
