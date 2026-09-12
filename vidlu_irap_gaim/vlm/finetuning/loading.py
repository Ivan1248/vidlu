"""Loading a fine-tuned VLM classifier from a training checkpoint, without an experiment.

`scripts/run.py test` reaches the fine-tuned model by rebuilding the whole experiment (data
factories, trainer, metrics).  For inference on arbitrary images and prompts none of that is
needed: a checkpoint's ``model_state.pth`` is exactly `_BaseVLMClassifier.state_dict()`, i.e.
``{"_config": ..., "_adapter_state": ...}``, and ``_config`` names everything required to
reconstruct the classifier.

Example::

    from vidlu_irap_gaim.vlm.finetuning import load_finetuned_classifier

    model = load_finetuned_classifier("~/data/experiments/states/<experiment>", which="best")
    response, thinking, truncated = model.generate_for_eval(image=pil_image, prompt="...")

The path of a checkpoint of a training run is printed by::

    python scripts/run.py get_checkpoint_path <data> <input_adapter> <model> <trainer> [other arguments]
"""

from pathlib import Path
import typing as T

import torch

from vidlu.training.checkpoint_manager import (WhichCheckpointArg, find_checkpoint_dir,
                                               get_file_name_and_interface)

from .model import _BaseVLMClassifier, Gemma4VLClassifier, Qwen3VLClassifier, Qwen35Classifier

# Checkpoints record the class name in `_config["classifier_class"]`.
CLASSIFIER_CLASSES: dict[str, type[_BaseVLMClassifier]] = {
    c.__name__: c for c in (Qwen3VLClassifier, Qwen35Classifier, Gemma4VLClassifier)
}

# Checkpoints written before `_get_config_dict` recorded the class name are identified by their
# base model instead: the classes declare distinct default model ids (`test_finetuned_loading`
# checks that), so a run that did not override `model_id` names its class unambiguously.
CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID: dict[str, type[_BaseVLMClassifier]] = {
    c._DEFAULT_MODEL_ID: c for c in CLASSIFIER_CLASSES.values()
}

# The model is in `separately_saved_state_parts` (see `vidlu.experiments.create_checkpoint_manager`),
# so it gets a file of its own instead of being packed into the rest of the training state.
MODEL_STATE_FILE_NAME, _MODEL_STATE_INTERFACE = get_file_name_and_interface("model_state")


def find_model_state_path(checkpoint: str | Path, which: WhichCheckpointArg = "best") -> Path:
    """Path of the model state file designated by `checkpoint`.

    Args:
        checkpoint: A `model_state.pth` file, a checkpoint directory containing one, or an
            experiment directory containing checkpoint directories.
        which: Which checkpoint to select if `checkpoint` is an experiment directory.

    Returns:
        The path of the `model_state.pth` file.
    """
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f'"{path}" is neither a file nor a directory.')
    if (state_path := path / MODEL_STATE_FILE_NAME).is_file():
        return state_path
    checkpoint_dir = find_checkpoint_dir(path, which=which)
    if not (state_path := checkpoint_dir / MODEL_STATE_FILE_NAME).is_file():
        raise FileNotFoundError(
            f'The checkpoint at "{checkpoint_dir}" contains no "{MODEL_STATE_FILE_NAME}". It was'
            + " probably not written by a VLM classifier experiment.")
    return state_path


def load_model_state(checkpoint: str | Path, which: WhichCheckpointArg = "best") -> dict:
    """Loads the `{"_config": ..., "_adapter_state": ...}` dictionary of a checkpoint to the CPU."""
    path = find_model_state_path(checkpoint, which=which)
    try:
        state = _MODEL_STATE_INTERFACE.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f'Could not load the model state at "{path}".') from e
    if not isinstance(state, T.Mapping) or "_config" not in state:
        raise ValueError(f'"{path}" is not the state of a VLM classifier: it has no "_config"'
                         + " entry. VLM classifiers store their configuration alongside the LoRA"
                         + " adapter weights, other models store parameters directly.")
    return state


def _name_the_class_hint() -> str:
    """The "say which class" advice both resolvers end their errors with.

    Only the advice is shared: the two diagnoses differ (a checkpoint that predates
    `classifier_class` versus a base model that names no class at all), and saying so
    is what makes either error actionable.
    """
    return ("Name the class explicitly (the `--classifier-class` option of the tools,"
            + f" `classifier_class` in Python): {sorted(CLASSIFIER_CLASSES)}.")


def resolve_classifier_class(class_name: str | None,
                             model_id: str | None) -> type[_BaseVLMClassifier]:
    """The classifier class a checkpoint's `_config` entry designates.

    Args:
        class_name: The `classifier_class` entry, absent in checkpoints written before it was
            recorded.
        model_id: The `model_id` entry, which identifies the class on its own whenever the
            training run did not override the class's default base model.

    Returns:
        The classifier class.
    """
    if class_name is not None:
        if (classifier_class := CLASSIFIER_CLASSES.get(class_name)) is None:
            raise ValueError(f'Unknown classifier class "{class_name}" in the checkpoint.'
                             + f" Known classes: {sorted(CLASSIFIER_CLASSES)}.")
        return classifier_class
    if (classifier_class := CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID.get(model_id)) is not None:
        print(f"[loading] The checkpoint predates `classifier_class`. Using"
              + f' {classifier_class.__name__}, whose default base model is "{model_id}".')
        return classifier_class
    raise ValueError(
        "The checkpoint does not record which classifier class wrote it (it predates"
        + f' `classifier_class` in `_get_config_dict`), and its base model "{model_id}" is not'
        + " the default of any known class. " + _name_the_class_hint())


def base_classifier_class_for(model_id: str | None) -> type[_BaseVLMClassifier]:
    """The classifier class to load a *pretrained* base model under.

    Unlike a checkpoint, a base model carries no record of its class, and the classes differ
    in chat template and processor -- so guessing wrong silently produces malformed prompts
    rather than an error. Only an unambiguous response is given.

    Args:
        model_id: The base model, or None for the default of `Qwen3VLClassifier`.

    Returns:
        The class whose default base model is `model_id`, or `Qwen3VLClassifier` when
        `model_id` is None.

    Raises:
        ValueError: When `model_id` is not the default base model of any known class, so
            the caller has to name the class.
    """
    if model_id is None:
        return Qwen3VLClassifier
    if (classifier_class := CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID.get(model_id)) is not None:
        return classifier_class
    raise ValueError(
        f'"{model_id}" is not the default base model of any known classifier class, so which'
        + " class to load it under is ambiguous. " + _name_the_class_hint())


def make_classifier_from_config(
        config: T.Mapping,
        *,
        classifier_class: type[_BaseVLMClassifier] | None = None,
        **overrides,
) -> _BaseVLMClassifier:
    """Constructs an unloaded classifier from a checkpoint's `_config` entry.

    Args:
        config: The `_config` entry of a checkpoint, as written by `_get_config_dict`.
        classifier_class: Overrides the class the checkpoint designates. Needed only for a
            checkpoint that predates `classifier_class` *and* overrode `model_id`.
        **overrides: Constructor arguments overriding or complementing those from `config`.

    Returns:
        The constructed classifier. The underlying HF model is not loaded yet.
    """
    kwargs = dict(config)
    class_name = kwargs.pop("classifier_class", None)
    if classifier_class is None:
        classifier_class = resolve_classifier_class(class_name, kwargs.get("model_id"))
    return classifier_class(**{**kwargs, **overrides})


def load_base_classifier(
        model_id: str | None = None,
        *,
        classifier_class: type[_BaseVLMClassifier] | None = None,
        device: str | torch.device = "cuda",
        **overrides,
) -> _BaseVLMClassifier:
    """Loads the pretrained model without fine-tuning, ready for generation.

    The counterpart of `load_finetuned_classifier` for measuring what the fine-tuning changed:
    the same prompting, image preprocessing and generation path, with the original weights. No
    LoRA adapter is attached at all, so this is the pretrained model itself rather than one with
    a zeroed adapter.

    Args:
        model_id: The base model, e.g. "Qwen/Qwen3-VL-8B-Instruct". None uses the default
            of `classifier_class`, or of `Qwen3VLClassifier` when that is None too.
        classifier_class: The model family to load. None infers it from `model_id`: the
            class whose default base model that is, or `Qwen3VLClassifier` when `model_id`
            is None too. An unrecognized `model_id` is an error rather than a guess -- the
            classes differ in their chat template and processor, so loading a model under
            the wrong one silently produces bad prompts.
        device: Device to load the model onto.
        **overrides: Further constructor arguments, e.g. `load_in_4bit`. Note that the default
            is 4-bit, matching what the fine-tuned checkpoints were trained against; pass
            `load_in_4bit=False` for the unquantized model.

    Returns:
        The classifier, loaded, in evaluation mode.

    Raises:
        ValueError: When `model_id` is not the default of any known class and no
            `classifier_class` says which one it is.
    """
    if classifier_class is None:
        classifier_class = base_classifier_class_for(model_id)
    model = classifier_class(
        model_id=model_id, use_lora=False, use_gradient_checkpointing=False, **overrides)
    model.to(device)  # before `initialize`, which loads the model onto `self._device`
    model.initialize(init_input=None)
    return model.eval()


def load_finetuned_classifier(
        checkpoint: str | Path,
        *,
        which: WhichCheckpointArg = "best",
        device: str | torch.device = "cuda",
        classifier_class: type[_BaseVLMClassifier] | None = None,
        **overrides,
) -> _BaseVLMClassifier:
    """Loads a fine-tuned VLM classifier from a training checkpoint, ready for generation.

    The base model and the LoRA configuration come from the checkpoint, so the caller does not
    have to repeat the arguments of the training command.

    Args:
        checkpoint: A `model_state.pth` file, a checkpoint directory containing one, or an
            experiment directory containing checkpoint directories.
        which: Which checkpoint to select if `checkpoint` is an experiment directory.
        device: Device to load the model onto.
        classifier_class: Overrides the class named by the checkpoint (see
            `make_classifier_from_config`).
        **overrides: Constructor arguments overriding those from the checkpoint, e.g.
            `load_in_4bit` or `enable_thinking`.

    Returns:
        The classifier, loaded, with the adapter weights applied, in evaluation mode.
    """
    state = load_model_state(checkpoint, which=which)
    model = make_classifier_from_config(
        state["_config"], classifier_class=classifier_class,
        # Gradient checkpointing only trades compute for memory during backpropagation.
        **{"use_gradient_checkpointing": False, **overrides})
    # Before `load_state_dict`, which loads the HF model, because `_load` reads `self._device`.
    model.to(device)
    model.load_state_dict(state)
    return model.eval()
