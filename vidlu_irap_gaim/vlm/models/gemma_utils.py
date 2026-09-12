"""
Shared utilities for Gemma chat message construction.

Used by zero-shot predictors and inference steps.
"""

from typing import Sequence

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


def build_gemma_generation_inputs(processor, pil_images: Sequence[Image.Image],
                                  prompts: Sequence[str], template_kwargs: dict):
    """Processor inputs to generate from, one conversation per (image, prompt) pair.

    The Gemma counterpart of ``qwen_utils.build_qwen_generation_inputs``, shared by
    the zero-shot predictor and the fine-tuned classifier.  ``apply_chat_template``
    takes a list of conversations and both tokenizes and pads them, so there is no
    separate processor call; the image is embedded in the messages (see
    `build_gemma_chat_messages` for why ``images=`` must not be passed as well).

    Padded on the left, since ``generate`` continues from the end of the sequence.

    Args:
        processor: The model's ``AutoProcessor``.
        pil_images: One image per conversation.
        prompts: One prompt per conversation.
        template_kwargs: Jinja variables for ``apply_chat_template`` (e.g.
            ``enable_thinking``).

    Returns:
        The processor's ``BatchEncoding``, on the CPU.
    """
    conversations = [build_gemma_chat_messages(pil_image, prompt)
                     for pil_image, prompt in zip(pil_images, prompts)]
    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
        **template_kwargs,
    )
