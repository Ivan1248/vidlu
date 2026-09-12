"""Answers custom prompts about images with a fine-tuned VLM checkpoint.

Unlike `vlm_inference.py`, which evaluates a whole dataset split with the iRAP attribute prompts,
this asks a fine-tuned model *one prompt of your choosing* about *one image of your choosing*. No
dataset, trainer or metrics are constructed; only the checkpoint is needed.

Single-shot::

    python -m vidlu_irap_gaim.tools.vlm_prompt --checkpoint <experiment-dir> \
        --image frame.jpg --prompt "Describe the roadside hazards."

Interactive (the model stays loaded between prompts, which matters because loading a 4-bit
Qwen3-VL takes minutes)::

    python -m vidlu_irap_gaim.tools.vlm_prompt --checkpoint <experiment-dir> --image frame.jpg

The pretrained model, without the fine-tuning, for comparison (no adapter is attached; the
prompting, preprocessing and generation are the same)::

    python -m vidlu_irap_gaim.tools.vlm_prompt --base-model --image frame.jpg --prompt "..."

The path of a checkpoint of a training run is printed by::

    python scripts/run.py get_checkpoint_path <data> <input_adapter> <model> <trainer> [other arguments]

Note that a LoRA-fine-tuned model is not a general-purpose assistant: it was trained on a single
fixed prompt (the one built by the response scheme and detail level of the training data), so
answers to unrelated prompts tend to drift towards the trained response format.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from vidlu_irap_gaim.vlm.finetuning.loading import (CLASSIFIER_CLASSES, find_model_state_path,
                                                    load_base_classifier,
                                                    load_finetuned_classifier)

DEFAULT_MAX_RESPONSE_TOKENS = 512

_INTERACTIVE_HELP = """Commands:
  :image <path>     use this image for subsequent prompts
  :file <path>      send the contents of a file as the prompt
  :tokens <n>       set the maximum number of response tokens ("none" to lift the limit)
  :thinking on|off  enable or disable the reasoning block
  :help             print this message
  :quit             exit (Ctrl-D also works)
Anything else is sent as a prompt about the current image."""


def load_image(path, upsampling_factor: float = 1) -> Image.Image:
    """Loads an image, optionally upsampling it as `VLMIrapDataset` does.

    The upsampling is bilinear here as well, but PIL's and `F.interpolate`'s bilinear filters
    differ slightly, so this reproduces the training-time resolution rather than exact pixels.
    """
    image = Image.open(path).convert("RGB")
    if upsampling_factor != 1:
        size = tuple(round(d * upsampling_factor) for d in image.size)
        image = image.resize(size, Image.BILINEAR)
    return image


def answer_prompt(model, image: Image.Image, prompt: str, *,
                  max_response_tokens: int | None = DEFAULT_MAX_RESPONSE_TOKENS,
                  min_new_tokens: int = 0) -> dict:
    """Generates an answer and returns it with its metadata as a JSON-ready dictionary.

    `max_response_tokens` of None lets the answer run until the model stops or the context
    window fills up. `amp` is off because the base weights already compute in bfloat16.
    """
    answer, thinking, is_truncated = model.generate_for_eval(
        image=image, prompt=prompt, max_response_tokens=max_response_tokens, amp=False,
        min_new_tokens=min_new_tokens)
    return dict(prompt=prompt, answer=answer, thinking=thinking, truncated=is_truncated)


def truncation_message(max_response_tokens: int | None) -> str:
    """Says why an answer was cut off, which differs when no budget was set."""
    return ("The answer filled the model's context window without ending."
            if max_response_tokens is None else
            f"The answer reached the {max_response_tokens}-token budget without ending.")


def report(record: dict, output_file=None, max_response_tokens: int | None = None):
    """Prints an `answer_prompt` result and optionally appends it to a JSON Lines file."""
    if (thinking := record["thinking"]) is not None:
        print(f"--- reasoning ---\n{thinking}")
    print(f"--- answer ---\n{record['answer']}")
    if record["truncated"]:
        remedy = ("" if max_response_tokens is None
                  else " Increase --max-response-tokens, or drop the limit with"
                       " --no-response-limit.")
        print(f"Warning: {truncation_message(max_response_tokens)}{remedy}", file=sys.stderr)
    if output_file is not None:
        output_file.write(json.dumps(record) + "\n")
        output_file.flush()


def _answer_current(model, image, image_path, prompt, max_response_tokens, min_new_tokens,
                    output_file):
    """Answers one interactive prompt about the currently selected image."""
    if image is None:
        print('No image selected. Use ":image <path>".', file=sys.stderr)
        return
    record = answer_prompt(model, image, prompt, max_response_tokens=max_response_tokens,
                           min_new_tokens=min_new_tokens)
    report(dict(image=str(image_path), **record), output_file, max_response_tokens)


def run_interactive(model, images: list[tuple[Path, Image.Image]], *,
                    max_response_tokens: int | None, min_new_tokens: int,
                    upsampling_factor: float, output_file=None):
    """Prompt loop over a model that stays loaded.

    Args:
        images: Initially available images; the last one is the current one.
        upsampling_factor: Applied to images loaded by the ":image" command.
    """
    image_path, image = images[-1] if images else (None, None)
    print(_INTERACTIVE_HELP)
    print(f"Current image: {image_path}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line == "":
            continue
        if not line.startswith(":"):
            _answer_current(model, image, image_path, line, max_response_tokens, min_new_tokens,
                            output_file)
            continue

        command, _, argument = line[1:].partition(" ")
        argument = argument.strip()
        if command in ("quit", "q", "exit"):
            return
        elif command == "help":
            print(_INTERACTIVE_HELP)
        elif command == "image":
            try:
                image = load_image(argument, upsampling_factor)
            except OSError as e:
                print(f"Could not load the image: {e}", file=sys.stderr)
            else:
                image_path = Path(argument)
                print(f"Current image: {image_path}")
        elif command == "tokens":
            if argument == "none":
                max_response_tokens = None
                print("No response-token limit beyond the context window.")
            elif argument.isdigit() and int(argument) > 0:
                max_response_tokens = int(argument)
                print(f"Maximum number of response tokens: {max_response_tokens}")
            else:
                print(f'"{argument}" is neither a positive integer nor "none".', file=sys.stderr)
        elif command == "thinking":
            if argument in ("on", "off"):
                model.enable_thinking = argument == "on"
                print(f"Thinking: {argument}")
            else:
                print('Expected "on" or "off".', file=sys.stderr)
        elif command == "file":
            try:
                prompt = Path(argument).read_text()
            except OSError as e:
                print(f"Could not read the prompt: {e}", file=sys.stderr)
            else:
                _answer_current(model, image, image_path, prompt, max_response_tokens,
                                min_new_tokens, output_file)
        else:
            print(f'Unknown command ":{command}". Type ":help" for the list.', file=sys.stderr)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Answers custom prompts about images with a fine-tuned VLM checkpoint.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=str, default=None,
                             help="Experiment directory, checkpoint directory, or model state"
                                  " file of a fine-tuned model.")
    model_group.add_argument("--base-model", action="store_true",
                             help="Use the pretrained model without the fine-tuning, for"
                                  " comparison. See also --model-id and --classifier-class.")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Base model for --base-model, e.g. \"Qwen/Qwen3-VL-8B-Instruct\"."
                             " Defaults to the classifier class's own default. A checkpoint"
                             " carries its own, so this cannot be combined with --checkpoint.")
    parser.add_argument("--which", type=str, choices=["best", "last"], default="best",
                        help="Which checkpoint to use if an experiment directory is given.")

    parser.add_argument("--image", type=str, nargs="+", default=[],
                        help="Image paths. In interactive mode, the last one is used first.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="The prompt. Without it (and without --prompt-file), the"
                                   + " prompts are read interactively.")
    prompt_group.add_argument("--prompt-file", type=str, default=None,
                              help="A file containing the prompt.")
    parser.add_argument("--max-response-tokens", type=int, default=DEFAULT_MAX_RESPONSE_TOKENS,
                        help="Maximum number of generated answer tokens.")
    parser.add_argument("--no-response-limit", action="store_true",
                        help="Ignore --max-response-tokens and generate until the model stops,"
                             " bounded only by the context window.")
    parser.add_argument("--min-new-tokens", type=int, default=0,
                        help="Minimum number of generated answer tokens (guards against an empty"
                             + " answer).")
    parser.add_argument("--upsampling-factor", type=float, default=1,
                        help="Image upsampling factor, matching `VLMIrapDataset`.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device.")
    parser.add_argument("--classifier-class", type=str, choices=sorted(CLASSIFIER_CLASSES),
                        default=None,
                        help="The classifier class that wrote the checkpoint. Needed only for a"
                             + " checkpoint that predates the recording of the class and does not"
                             + " use its class's default base model.")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None,
                        help="Overrides the quantization the checkpoint was trained with.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                        help="Overrides whether a reasoning block is requested.")
    parser.add_argument("--output", type=str, default=None,
                        help="A JSON Lines file to append the answers to.")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    max_response_tokens = None if args.no_response_limit else args.max_response_tokens
    prompt = (Path(args.prompt_file).read_text() if args.prompt_file is not None else args.prompt)
    if prompt is not None and len(args.image) == 0:
        parser.error("--prompt and --prompt-file require at least one --image.")
    if args.model_id is not None and not args.base_model:
        parser.error("--model-id applies to --base-model; a checkpoint records its own.")

    images = [(Path(p), load_image(p, args.upsampling_factor)) for p in args.image]

    overrides = {name: value for name, value in
                 dict(load_in_4bit=args.load_in_4bit, enable_thinking=args.thinking).items()
                 if value is not None}
    classifier_class = (None if args.classifier_class is None
                        else CLASSIFIER_CLASSES[args.classifier_class])
    if args.base_model:
        state_path = None
        print("Loading the pretrained model, without the fine-tuning...")
        model = load_base_classifier(args.model_id, classifier_class=classifier_class,
                                     device=args.device, **overrides)
    else:
        # Resolved here rather than inside the loader so that the answers record the exact
        # checkpoint.
        state_path = find_model_state_path(args.checkpoint, which=args.which)
        print(f"Loading the checkpoint at {state_path.parent}...")
        model = load_finetuned_classifier(state_path, device=args.device,
                                          classifier_class=classifier_class, **overrides)

    output_file = open(args.output, "a") if args.output is not None else None
    try:
        if prompt is None:
            run_interactive(model, images, max_response_tokens=max_response_tokens,
                            min_new_tokens=args.min_new_tokens,
                            upsampling_factor=args.upsampling_factor, output_file=output_file)
        else:
            for image_path, image in images:
                print(f"\n=== {image_path} ===")
                record = answer_prompt(model, image, prompt,
                                       max_response_tokens=max_response_tokens,
                                       min_new_tokens=args.min_new_tokens)
                report(dict(checkpoint=None if state_path is None else str(state_path),
                            model_id=model.model_id, image=str(image_path), **record),
                       output_file, max_response_tokens)
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
