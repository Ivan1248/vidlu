"""
Shared utilities for Qwen-VL chat message construction.

Used by zero-shot predictors, fine-tuned inference, and training steps.
"""

from typing import Sequence

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


def build_qwen_generation_inputs(processor, pil_images: Sequence[Image.Image],
                                 prompts: Sequence[str], template_kwargs: dict):
    """Processor inputs to generate from, one row per (image, prompt) pair.

    The single place that renders the generation prompt and preprocesses the images
    for the Qwen-VL family, shared by the zero-shot predictor and the fine-tuned
    classifier so that the two cannot prompt differently.

    Padded on the left: ``generate`` continues from the end of the sequence, so
    right padding would put pad tokens where the response starts.  A batch with a
    fixed prompt and equal-size images has equal-length rows and gets no padding.

    Args:
        processor: The model's ``AutoProcessor``.
        pil_images: One image per row.
        prompts: One prompt per row.
        template_kwargs: Jinja variables for ``apply_chat_template`` (e.g.
            ``enable_thinking``).

    Returns:
        The processor's ``BatchEncoding``, on the CPU.
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    messages_batch = [build_qwen_chat_messages(pil_image, prompt)
                      for pil_image, prompt in zip(pil_images, prompts)]
    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **template_kwargs)
             for messages in messages_batch]
    image_inputs, video_inputs = [], []
    for messages in messages_batch:
        images, videos = process_vision_info(messages)
        image_inputs.extend(images or [])
        video_inputs.extend(videos or [])
    return processor(
        text=texts,
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
