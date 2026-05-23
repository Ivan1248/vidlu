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

    Gemma 4 (transformers >= 5.5) expects the actual PIL image embedded
    in the message content as ``{"type": "image", "image": <pil>}``.  The
    processor extracts images from messages on its own; passing
    ``images=[pil]`` to ``apply_chat_template`` in addition raises
    ``TypeError: ... got multiple values for keyword argument 'images'``.
    Images come before text.

    Args:
        pil_image: Input image as PIL Image (embedded in the message).
        prompt: User prompt text.
        response: Optional assistant response to append (for training sequences).

    Returns:
        List of message dicts in Gemma chat format.
    """
    content = [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
