"""Tests for `_BaseVLMClassifier.generate_batch_for_eval`.

Per-attribute evaluation asks 41 separate questions per image, so generating one
row at a time is what makes it expensive: decoding a dozen tokens is
latency-bound, not compute-bound. Batching is therefore load-bearing, and it is
also the part most easily wrong in a way that shows up as a *quality* difference
rather than a crash – padding on the wrong side, or a truncation flag read off
the wrong row, would silently degrade one arm of the comparison it exists to
serve.

Pure unit tests: a stub HF-like model that emits scripted token sequences per
row, plus a stub processor. No weights, no GPU. The real end-to-end equivalence
check against single-row generation needs a model and is opt-in; see
`test_batched_matches_single_row_on_a_real_model`.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vidlu_irap_gaim.vlm.finetuning.model import _BaseVLMClassifier

VOCAB = ["<pad>", "1:", "Urban", "Rural", "Commercial", "<eos>"]
ID = {tok: i for i, tok in enumerate(VOCAB)}
CONTEXT_LENGTH = 64


class _StubTokenizer:
    eos_token_id = ID["<eos>"]
    pad_token_id = ID["<pad>"]

    def encode(self, text, add_special_tokens=False):
        return [ID[t] for t in text.split()]


class _StubProcessor:
    tokenizer = _StubTokenizer()

    def batch_decode(self, ids, skip_special_tokens=True):
        skipped = {ID["<pad>"], ID["<eos>"]} if skip_special_tokens else set()
        return [" ".join(VOCAB[int(i)] for i in row if int(i) not in skipped)
                for row in ids]


class _ScriptedModel:
    """Emits a preset continuation per row, capped at max_new_tokens.

    Rows that stop early are padded out to the longest row, exactly as
    ``generate`` does – which is why truncation cannot be read off the last
    position of a batched result.
    """

    generation_config = None

    def __init__(self, scripts_per_call):
        self.scripts_per_call = list(scripts_per_call)
        self.calls = []
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=CONTEXT_LENGTH))

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, **kwargs):
        scripts = self.scripts_per_call.pop(0)
        self.calls.append({"batch_size": input_ids.shape[0],
                           "prompt_len": input_ids.shape[1],
                           "max_new_tokens": max_new_tokens})
        emitted = [s[:max_new_tokens] for s in scripts]
        width = max(len(e) for e in emitted)
        rows = [[ID[t] for t in e] + [ID["<pad>"]] * (width - len(e)) for e in emitted]
        new = torch.tensor(rows, dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


class _Classifier(_BaseVLMClassifier):
    """Exposes batched generation over the stubs.

    ``_build_generation_inputs`` is stubbed to record what it was asked for and
    to produce the attention mask the caller wants, since raggedness is what
    decides whether batching is safe.
    """

    def __init__(self, scripts_per_call, *, enable_thinking=None, prompt_lengths=None,
                 supports_left_padding=True):
        super().__init__(enable_thinking=enable_thinking)
        self._model = _ScriptedModel(scripts_per_call)
        self._processor = _StubProcessor()
        self._prompt_lengths = prompt_lengths
        self._SUPPORTS_LEFT_PADDED_GENERATION = supports_left_padding
        self.build_calls = []

    def _build_generation_inputs(self, pil_images, prompts):
        self.build_calls.append({"images": list(pil_images), "prompts": list(prompts)})
        lengths = self._prompt_lengths or [4] * len(prompts)
        lengths = lengths[:len(prompts)]
        width = max(lengths)
        input_ids = torch.zeros(len(prompts), width, dtype=torch.long)
        # Left padding: the real positions sit at the end of each row.
        attention_mask = torch.zeros(len(prompts), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            attention_mask[row, width - length:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _images(n):
    """Distinguishable stand-in images; `to_pil_image` accepts (C, H, W) tensors."""
    return [torch.full((3, 4, 4), i / 10.0) for i in range(n)]


# ----- batching ---------------------------------------------------------------

def test_the_whole_batch_uses_one_generate_call():
    model = _Classifier([[["1:", "Urban", "<eos>"],
                          ["1:", "Rural", "<eos>"],
                          ["1:", "Commercial", "<eos>"]]])

    results = model.generate_batch_for_eval(_images(3), ["q"] * 3, max_response_tokens=8)

    assert [response for response, _, _ in results] == ["1: Urban", "1: Rural", "1: Commercial"]
    assert len(model._model.calls) == 1
    assert model._model.calls[0]["batch_size"] == 3


def test_generate_for_eval_is_the_one_element_case():
    """One generation path, so evaluation during and after training cannot
    diverge – and so the batched path is exercised by every existing caller."""
    model = _Classifier([[["1:", "Urban", "<eos>"]]])

    response, thinking, is_truncated = model.generate_for_eval(_images(1)[0], "q",
                                                            max_response_tokens=8)

    assert (response, thinking, is_truncated) == ("1: Urban", None, False)
    assert model._model.calls[0]["batch_size"] == 1


def test_images_and_prompts_are_paired_positionally():
    model = _Classifier([[["1:", "Urban", "<eos>"], ["1:", "Rural", "<eos>"]]])

    model.generate_batch_for_eval(_images(2), ["a", "b"], max_response_tokens=8)

    assert len(model.build_calls) == 1
    assert model.build_calls[0]["prompts"] == ["a", "b"]
    assert len(model.build_calls[0]["images"]) == 2


def test_mismatched_lengths_raise():
    model = _Classifier([])
    with pytest.raises(ValueError, match="2 images but 1 prompts"):
        model.generate_batch_for_eval(_images(2), ["only one"])


def test_an_empty_batch_generates_nothing():
    model = _Classifier([])
    assert model.generate_batch_for_eval([], []) == []
    assert model._model.calls == []


# ----- truncation -------------------------------------------------------------

def test_truncation_is_per_row():
    """A row that stopped on its own is complete even though the batch ran to the
    cap for another row. Reading the last position instead would call it
    truncated, because `generate` pads the finished row out."""
    model = _Classifier([[["1:", "Urban", "<eos>"],  # stops on its own
                          ["1:", "Rural", "Rural", "Rural"]]])  # runs into the cap

    results = model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=4)

    assert [truncated for _, _, truncated in results] == [False, True]


def test_nothing_is_truncated_below_the_cap():
    model = _Classifier([[["1:", "Urban", "<eos>"], ["1:", "Rural", "<eos>"]]])

    results = model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=16)

    assert [truncated for _, _, truncated in results] == [False, False]


# ----- falling back rather than risking a wrong response -------------------------

def test_a_ragged_batch_falls_back_when_the_family_cannot_be_left_padded():
    """Qwen3.5's recurrence is only correct with right padding, but generation
    needs left padding, so a ragged batch there has to be generated row by row."""
    model = _Classifier([[["1:", "Urban", "<eos>"]], [["1:", "Rural", "<eos>"]]],
                        prompt_lengths=[4, 6], supports_left_padding=False)

    results = model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=8)

    assert [response for response, _, _ in results] == ["1: Urban", "1: Rural"]
    assert [call["batch_size"] for call in model._model.calls] == [1, 1]


def test_an_equal_length_batch_is_batched_even_without_left_padding_support():
    """Equal-length rows get no padding at all, so the constraint does not bite.
    Holding the prompt fixed and varying the image is what makes them equal."""
    model = _Classifier([[["1:", "Urban", "<eos>"], ["1:", "Rural", "<eos>"]]],
                        prompt_lengths=[4, 4], supports_left_padding=False)

    model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=8)

    assert [call["batch_size"] for call in model._model.calls] == [2]


def test_a_ragged_batch_is_batched_when_the_family_can_be_left_padded():
    model = _Classifier([[["1:", "Urban", "<eos>"], ["1:", "Rural", "<eos>"]]],
                        prompt_lengths=[4, 6], supports_left_padding=True)

    model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=8)

    assert [call["batch_size"] for call in model._model.calls] == [2]


def test_thinking_generates_row_by_row():
    """The reasoning/response budget split forces each sequence's block closed
    individually, so it cannot be shared across a batch."""
    model = _Classifier([[["1:", "Urban", "<eos>"]], [["1:", "Rural", "<eos>"]]],
                        enable_thinking=True)

    results = model.generate_batch_for_eval(_images(2), ["q"] * 2, max_response_tokens=8)

    assert len(results) == 2
    assert [call["batch_size"] for call in model._model.calls] == [1, 1]


# ----- real model (opt-in) -----------------------------------------------------
#
# What these must NOT assert: that batched and single-row generation produce
# byte-identical free-form text.  A batched matmul reduces in a different order
# and picks different kernels than a batch of one, so logits differ in their last
# bits; on an open-ended task that flips the argmax at some near-tied token and
# the continuations diverge from there.  An earlier version of this file asserted
# exactly that and failed on two fluent paraphrases of the same correct response --
# measuring floating-point noise, not correctness.  Worse, it batched
# equal-length rows, so it never inserted a pad token and never tested the
# padding it existed to test.
#
# So: check the padding directly, on logits, where a tolerance is meaningful; and
# check end-to-end agreement on the *constrained* output the evaluation actually
# parses, where the response space is closed and ties are not a coin flip.

REAL_MODEL_ONLY = pytest.mark.skipif(
    not os.environ.get("VLM_TEST_MODEL_ID"),
    reason="Set VLM_TEST_MODEL_ID to run against a real model")

# Bound on how far a padded row's logits may move, as a fraction of that row's
# logit range.  Relative rather than absolute so it carries across models and
# quantizations: an absolute bound calibrated on a 4-bit checkpoint is far too
# tight for one whose noise floor is lower, and too loose for one whose logits
# are compressed.
#
# Measured on Qwen3-VL-8B-Instruct in 4-bit NF4: max |difference| 0.59 and 0.35
# over a ~150k-token vocabulary, a few percent of the logit range.  bitsandbytes
# blockwise dequantization and bf16 matmuls both pick different kernels at
# different batch shapes, so that is the noise floor, not a defect.
#
# This is the secondary check.  The sharp one is the argmax: broken padding
# shifts position ids or lets the model attend to pad tokens, which moves logits
# by far more than this and changes which token decodes.
MAX_RELATIVE_LOGIT_DIFF = 0.1


@pytest.fixture(scope="module")
def real_classifier():
    from vidlu_irap_gaim.vlm.finetuning.loading import (CLASSIFIER_CLASSES,
                                                        load_base_classifier)

    # The class is inferred from the model id, which works for the classes' default
    # base models; name it for anything else (a fork, or a local path).
    class_name = os.environ.get("VLM_TEST_CLASSIFIER_CLASS")
    return load_base_classifier(
        os.environ["VLM_TEST_MODEL_ID"],
        classifier_class=None if class_name is None else CLASSIFIER_CLASSES[class_name])


@REAL_MODEL_ONLY
def test_a_padded_batch_matches_each_row_run_alone(real_classifier):
    """The check the stubs cannot make: that left padding is handled correctly.

    Generation continues from the end of the sequence, so the rows of a ragged
    batch have to be padded on the *left* – and Qwen3-VL derives its M-RoPE
    position ids from the attention mask, so getting either wrong silently
    degrades the padded rows rather than raising.

    Comparing next-token logits rather than generated text keeps the comparison
    on one forward pass, where the only difference between the two is the padding
    itself and a tolerance therefore means something.
    """
    import torch
    from PIL import Image

    classifier = real_classifier
    images = [Image.new("RGB", (224, 224), color=(10, 20, 30)),
              Image.new("RGB", (224, 224), color=(200, 180, 160))]
    # Deliberately different lengths, so the batch actually needs padding.
    prompts = ["What colour is this?",
               "Describe this image in detail, mentioning the colour, the shape, "
               "and anything else that stands out to a careful observer."]

    inputs = classifier._build_generation_inputs(images, prompts)
    assert not bool(inputs["attention_mask"].all()), (
        "This batch was padding-free, so it cannot test padding. Make the prompt "
        "lengths differ more.")

    with torch.no_grad():
        batched_logits = classifier._model(**inputs).logits[:, -1, :].float()

    for row, (image, prompt) in enumerate(zip(images, prompts)):
        solo_inputs = classifier._build_generation_inputs([image], [prompt])
        assert bool(solo_inputs["attention_mask"].all()), "a single row needs no padding"
        with torch.no_grad():
            solo_logits = classifier._model(**solo_inputs).logits[:, -1, :].float()

        difference = (batched_logits[row] - solo_logits[0]).abs().max().item()
        logit_range = (solo_logits[0].max() - solo_logits[0].min()).item()
        relative = difference / logit_range
        print(f"row {row}: max |logit difference| = {difference:.4f}, "
              f"logit range = {logit_range:.2f}, relative = {relative:.4f}")

        assert batched_logits[row].argmax() == solo_logits[0].argmax(), (
            f"Row {row} would decode a different next token when padded "
            f"(max |difference| {difference:.4f}, {relative:.1%} of its logit range). "
            f"Padding or position ids are wrong.")
        assert relative < MAX_RELATIVE_LOGIT_DIFF, (
            f"Row {row} logits moved by {difference:.4f} when padded, {relative:.1%} of "
            f"its {logit_range:.2f} range, beyond the {MAX_RELATIVE_LOGIT_DIFF:.0%} "
            f"allowed for batching noise.")


@REAL_MODEL_ONLY
def test_batched_and_single_row_parse_to_the_same_response(real_classifier):
    """End-to-end agreement on the output the evaluation actually scores.

    The metric never sees the response text, only the class index parsed out of
    it. That output space is closed and the format is fixed, so unlike free-form
    text it does not hinge on a near-tied continuation – which makes exact
    agreement the right assertion here and the wrong one there.
    """
    from PIL import Image

    from vidlu_irap_gaim.vlm.response_scheme import make_response_scheme

    attr = "Area type"
    metadata = {attr: {"Urban": 0, "Rural": 1}}
    scheme = make_response_scheme("standard", metadata)
    prompt = scheme.build_prompt([attr])

    classifier = real_classifier
    images = [Image.new("RGB", (224, 224), color=c)
              for c in ((10, 20, 30), (200, 180, 160), (90, 90, 90))]
    budget = scheme.compute_max_response_tokens(classifier.tokenizer, [attr], margin_tokens=32)

    batched = classifier.generate_batch_for_eval(
        images, [prompt] * len(images), max_response_tokens=budget)
    one_by_one = [classifier.generate_for_eval(image, prompt, max_response_tokens=budget)
                  for image in images]

    def parsed(results):
        return [scheme.parse_response(response, [attr])[attr].pred_idx
                for response, _, _ in results]

    assert parsed(batched) == parsed(one_by_one), (
        f"batched responses {[a for a, _, _ in batched]} vs "
        f"single-row {[a for a, _, _ in one_by_one]}")
