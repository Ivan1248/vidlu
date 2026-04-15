"""
Shared utilities for Gemma chat message construction.

Used by zero-shot predictors and inference steps.
"""

from PIL import Image


def build_gemma_chat_messages(
    pil_image: Image.Image,
    prompt: str,
    response: str | None = None,
) -> list[dict]:
    """Build Gemma chat message list for processor.apply_chat_template.

    Gemma 4 expects image placeholders in the message content, with actual
    PIL images passed separately to the processor. Images come before text.

    Args:
        pil_image: Input image as PIL Image (passed to processor separately).
        prompt: User prompt text.
        response: Optional assistant response to append (for training sequences).

    Returns:
        List of message dicts in Gemma chat format.
    """
    content = [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
