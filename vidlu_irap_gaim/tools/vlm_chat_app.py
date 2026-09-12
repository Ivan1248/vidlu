"""Streamlit chat page for a fine-tuned VLM checkpoint.

The browser counterpart of `vlm_prompt.py`: upload an image, type a question, read the answer.
Run it with::

    streamlit run vidlu_irap_gaim/tools/vlm_chat_app.py -- --checkpoint <experiment-dir>

The sidebar switches between a fine-tuned checkpoint and the pretrained model (`--base-model`
starts on the latter), so the two can be compared on the same image and question. It also lifts
the response-token limit, for answers that need more than the default budget.

The arguments after `--` are optional; the checkpoint can also be entered in the sidebar. The
path of a checkpoint of a training run is printed by::

    python scripts/run.py get_checkpoint_path <data> <input_adapter> <model> <trainer> [other arguments]

The page is served on port 8501 of whatever the process can see. Two things commonly hide it:

- In a container, the port must be published or the container must share the host network;
  otherwise the host cannot reach it even though the container reports `0.0.0.0:8501`.
  `scripts/podman.sh` runs with `--network=host`, so nothing more is needed there; with plain
  docker/podman, pass `-p 8501:8501` (or `--network=host`). Do not pass
  `--server.address 127.0.0.1` in a container without host networking -- that binds the
  container's loopback interface, which nothing outside it can reach. Rootless podman also
  makes the container's IP unroutable from the host, so there is no third way in.
- From another machine, forward the port over SSH::

    ssh -L 8501:localhost:8501 <host>

Each message is answered independently: `generate_for_eval` builds a single-turn conversation
(one image, one question), so earlier messages are history for the reader, not context for the
model. Multi-turn conversation would need a message-sequence generation method on the classifier.
"""

import argparse
from pathlib import Path
import sys

# `streamlit run <path>` puts the script's own directory on `sys.path`, not the working directory
# (which is what `python -m` would use), so the repository root has to be added for `vidlu` and
# `vidlu_irap_gaim` to be importable without installing them. This mirrors `scripts/_context.py`,
# which does the same for `run.py`; `irap-data` is handled by `vidlu_irap_gaim/__init__.py`.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from vidlu_irap_gaim.tools.vlm_prompt import (DEFAULT_MAX_RESPONSE_TOKENS,  # noqa: E402
                                              answer_prompt, truncation_message)
from vidlu_irap_gaim.vlm.finetuning.loading import (CLASSIFIER_CLASSES,  # noqa: E402
                                                    find_model_state_path,
                                                    load_base_classifier,
                                                    load_finetuned_classifier)

IMAGE_FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

FINE_TUNED, PRETRAINED = "Fine-tuned checkpoint", "Pretrained, not fine-tuned"

# `st.chat_input(accept_file=...)`, i.e. an image and a question in one message box.
MIN_STREAMLIT_VERSION = (1, 43)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streamlit chat page for a fine-tuned VLM checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Experiment directory, checkpoint directory, or model state file.")
    parser.add_argument("--which", type=str, choices=["best", "last"], default="best",
                        help="Which checkpoint to use if an experiment directory is given.")
    parser.add_argument("--base-model", action="store_true",
                        help="Start on the pretrained model instead of a fine-tuned checkpoint.")
    parser.add_argument("--model-id", type=str, default="",
                        help="Base model for --base-model. Empty means the classifier class's"
                             " own default.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device.")
    parser.add_argument("--classifier-class", type=str, choices=sorted(CLASSIFIER_CLASSES),
                        default=None,
                        help="The classifier class that wrote the checkpoint. Needed only for a"
                             " checkpoint that predates the recording of the class and does not"
                             " use its class's default base model.")
    # Strict: `streamlit run app.py -- <args>` passes exactly these arguments, so an unknown one
    # is a mistake worth reporting rather than ignoring.
    return parser.parse_args(argv)


@st.cache_resource(show_spinner=False, hash_funcs={Path: str})
def load_model(state_path: Path | None, model_id: str | None, device: str,
               classifier_class_name: str | None = None):
    """Loads a classifier once per distinct model; Streamlit reruns reuse it.

    A `state_path` loads that fine-tuned checkpoint; without one, the pretrained `model_id` is
    loaded unmodified. The class is passed by name rather than as a type so that it is part of
    the cache key.
    """
    classifier_class = (None if classifier_class_name is None
                        else CLASSIFIER_CLASSES[classifier_class_name])
    if state_path is None:
        return load_base_classifier(model_id or None, classifier_class=classifier_class,
                                    device=device)
    return load_finetuned_classifier(state_path, device=device,
                                     classifier_class=classifier_class)


def sidebar_controls(defaults: argparse.Namespace) -> dict:
    """Renders the sidebar and returns the checkpoint selection and generation settings."""
    st.sidebar.header("Model")
    source = st.sidebar.radio(
        "Weights", [FINE_TUNED, PRETRAINED], index=1 if defaults.base_model else 0,
        help="The pretrained model is the same model without the fine-tuning, for comparison.")
    checkpoint, which, model_id = "", defaults.which, ""
    if source == FINE_TUNED:
        checkpoint = st.sidebar.text_input(
            "Experiment or checkpoint directory", value=defaults.checkpoint,
            help="An experiment directory, a checkpoint directory, or a model state file.")
        which = st.sidebar.radio("Checkpoint to use", ["best", "last"],
                                 index=["best", "last"].index(defaults.which), horizontal=True,
                                 help="Used only when an experiment directory is given.")
    else:
        model_id = st.sidebar.text_input(
            "Base model", value=defaults.model_id,
            placeholder="empty: the classifier class's default",
            help="A Hugging Face model id, e.g. \"Qwen/Qwen3-VL-8B-Instruct\".")

    st.sidebar.header("Generation")
    no_response_limit = st.sidebar.checkbox(
        "No response-token limit", value=False,
        help="Generate until the model stops on its own. The context window is still the"
             " ceiling, and a long answer takes correspondingly long.")
    max_response_tokens = st.sidebar.number_input(
        "Maximum response tokens", min_value=16, max_value=8192,
        value=DEFAULT_MAX_RESPONSE_TOKENS, step=16, disabled=no_response_limit)
    min_new_tokens = st.sidebar.number_input(
        "Minimum response tokens", min_value=0, max_value=1024, value=0, step=1,
        help="Guards against an empty answer.")
    upsampling_factor = st.sidebar.number_input(
        "Image upsampling factor", min_value=0.25, max_value=4.0, value=1.0, step=0.25,
        help="Matches the `upsampling_factor` of `VLMIrapDataset`.")
    return dict(checkpoint=checkpoint, which=which, model_id=model_id,
                is_fine_tuned=source == FINE_TUNED,
                max_response_tokens=None if no_response_limit else int(max_response_tokens),
                min_new_tokens=int(min_new_tokens), upsampling_factor=float(upsampling_factor))


def upsample(image: Image.Image, factor: float) -> Image.Image:
    if factor == 1:
        return image
    return image.resize(tuple(round(d * factor) for d in image.size), Image.BILINEAR)


def render_history():
    """Renders the messages exchanged so far. They are not fed back to the model."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if (image := message.get("image")) is not None:
                st.image(image, width=360)
            if (thinking := message.get("thinking")) is not None:
                with st.expander("Reasoning"):
                    st.markdown(thinking)
            st.markdown(message["content"])
            if (truncation := message.get("truncation")) is not None:
                st.warning(truncation)


def check_streamlit_version():
    version = tuple(int(p) for p in st.__version__.split(".")[:2])
    if version < MIN_STREAMLIT_VERSION:
        raise RuntimeError(
            f"Streamlit {'.'.join(map(str, MIN_STREAMLIT_VERSION))} or newer is required for"
            f" image attachments in the message box, but {st.__version__} is installed.")


def main():
    check_streamlit_version()
    defaults = parse_args(sys.argv[1:])
    st.set_page_config(page_title="Fine-tuned VLM chat", layout="centered")
    st.title("Fine-tuned VLM chat")

    settings = sidebar_controls(defaults)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("image", None)

    state_path = None
    if settings["is_fine_tuned"]:
        if not settings["checkpoint"]:
            st.info("Enter a checkpoint or experiment directory in the sidebar.")
            return
        try:
            state_path = find_model_state_path(settings["checkpoint"], which=settings["which"])
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            return

    description = ("the pretrained model" if state_path is None
                   else f"the checkpoint at {state_path.parent}")
    with st.spinner(f"Loading {description}..."):
        model = load_model(state_path, settings["model_id"], defaults.device,
                           defaults.classifier_class)
    st.caption(f"{type(model).__name__} ({model.model_id}), {description}."
               " Each question is answered on its own; the transcript is not sent to the model.")

    if model.enable_thinking is not None:
        # The model is cached across reruns, so this setting is shared by all browser sessions.
        # `None` means "leave the chat template's default alone", which a toggle cannot express.
        model.enable_thinking = st.sidebar.toggle("Thinking", value=model.enable_thinking,
                                                  help="Request a reasoning block.")

    render_history()

    submission = st.chat_input("Ask something about the image", accept_file=True,
                               file_type=IMAGE_FILE_TYPES)
    if submission is None:
        return

    # `st.chat_input` with `accept_file` returns an object with `.text` and `.files`.
    if submission.files:
        st.session_state.image = Image.open(submission.files[-1]).convert("RGB")
    prompt, image = submission.text, st.session_state.image
    if image is None:
        st.warning("Attach an image first (the paper-clip button in the message box).")
        return
    if not prompt:
        return

    st.session_state.messages.append(
        dict(role="user", content=prompt, image=image if submission.files else None))
    with st.chat_message("user"):
        if submission.files:
            st.image(image, width=360)
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Generating..."):
        record = answer_prompt(model, upsample(image, settings["upsampling_factor"]), prompt,
                               max_response_tokens=settings["max_response_tokens"],
                               min_new_tokens=settings["min_new_tokens"])
        if record["thinking"] is not None:
            with st.expander("Reasoning"):
                st.markdown(record["thinking"])
        st.markdown(record["answer"])
        truncation = None
        if record["truncated"]:
            remedy = ("" if settings["max_response_tokens"] is None
                      else ' Raise "Maximum response tokens", or tick "No response-token limit".')
            truncation = truncation_message(settings["max_response_tokens"]) + remedy
            st.warning(truncation)
    # The message keeps its own warning: the sidebar may say something else by the time the
    # transcript is re-rendered.
    st.session_state.messages.append(
        dict(role="assistant", content=record["answer"], thinking=record["thinking"],
             truncation=truncation))


if __name__ == "__main__":
    main()
