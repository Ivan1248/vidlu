"""
Utilities for separating thinking/reasoning blocks from VLM responses.

Each model family uses a different delimiter for thinking blocks.
Add new patterns here when integrating additional model families.
"""

# Extra tokens reserved for reasoning on top of the response budget when thinking
# is enabled.  Shared by the inference CLI and the fine-tuning eval step so the
# two cannot drift apart.
DEFAULT_THINKING_BUDGET = 4096

# Closing delimiter per model family:
#   Qwen3 / Qwen3.5: <think>...</think>
#   Gemma 4:         <|channel>thought\n...\n<channel|>
QWEN_THINKING_END = "</think>"
GEMMA_THINKING_END = "<channel|>"

# Only the *closing* delimiter is matched.  With ``add_generation_prompt=True``
# the chat template emits the *opening* delimiter into the prompt (verified in
# Qwen/Qwen3.5-9B's chat template), so the model generates only the closing one.
# Requiring a full `<open>...<close>` pair therefore fails to match on exactly
# the models that reason.
#
# Read-side matching stays permissive across families (one helper serves every
# predictor); *forcing* a close needs the one delimiter this model actually uses,
# which is why each classifier names its own.
_THINKING_END_DELIMITERS = (QWEN_THINKING_END, GEMMA_THINKING_END)


def has_thinking_end(text: str) -> bool:
    """Whether a closing thinking delimiter is present.

    False on a response truncated mid-reasoning – the case where the reasoning
    would otherwise be mistaken for the response.
    """
    return any(d in text for d in _THINKING_END_DELIMITERS)


def split_thinking(raw_response: str) -> tuple[str, str | None]:
    """Splits generated text into its response and its thinking block.

    The thinking block is everything up to and including the first closing
    delimiter, whether or not the opening delimiter is present.  One rule covers
    both layouts: a model that emits ``<think>reasoning</think>`` in full, and
    one whose opening tag came from the generation prompt so that it emits only
    ``reasoning</think>``.

    Returns:
        Tuple of (response, thinking_text).  *thinking_text* is None when no
        closing delimiter was found – which is also the correct result when
        thinking is disabled, and when generation was truncated mid-reasoning
        (the caller then sees the reasoning as the response, rather than the
        reasoning being silently dropped along with the missing response).
    """
    found = [(i, d) for d in _THINKING_END_DELIMITERS if (i := raw_response.find(d)) != -1]
    if not found:
        return raw_response, None
    start, delimiter = min(found)  # the earliest closing delimiter
    end = start + len(delimiter)
    return raw_response[end:].strip(), raw_response[:end]
