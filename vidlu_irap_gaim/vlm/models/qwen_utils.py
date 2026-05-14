"""
Shared utilities for Qwen-VL chat message construction.

Used by zero-shot predictors, fine-tuned inference, and training steps.
"""

from PIL import Image


def build_qwen_chat_messages(
    pil_image: Image.Image,
    prompt: str,
    response: str | None = None,
) -> list[dict]:
    """Build Qwen-VL chat message list for processor/process_vision_info.

    Args:
        pil_image: Input image as PIL Image.
        prompt: User prompt text.
        response: Optional assistant response to append (for training sequences).

    Returns:
        List of message dicts in Qwen-VL format.
    """
    content = [
        {"type": "text", "text": prompt},
        {"type": "image", "image": pil_image},
    ]
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
