"""
Utilities for stripping thinking/reasoning blocks from VLM responses.

Each model family uses a different delimiter for thinking blocks.
Add new patterns here when integrating additional model families.
"""

import re

# Qwen3: <think>...</think>
_QWEN_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Gemma 4: <|channel>thought\n...\n<channel|>
_GEMMA_THINK_RE = re.compile(r"<\|channel>thought\n.*?\n<channel\|>", re.DOTALL)


def strip_thinking(raw_response: str) -> tuple[str, str | None]:
    """Remove thinking/reasoning blocks from a VLM response.

    Handles both Qwen3 (``<think>...</think>``) and Gemma 4
    (``<|channel>thought\\n...\\n<channel|>``) formats.

    Returns:
        Tuple of (clean_response, thinking_text).  *thinking_text* is None
        when no thinking block was found.
    """
    for pattern in (_QWEN_THINK_RE, _GEMMA_THINK_RE):
        match = pattern.search(raw_response)
        if match:
            thinking_text = match.group(0)
            clean = raw_response[:match.start()] + raw_response[match.end():]
            return clean.strip(), thinking_text
    return raw_response, None
