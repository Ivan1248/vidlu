"""Tests for loading a fine-tuned VLM classifier from a training checkpoint.

Covers the checkpoint-selection rules and the reconstruction of a classifier from the `_config`
entry a checkpoint stores, so that inference tools do not have to repeat the arguments of the
training command.

Pure unit tests: `_load` (which would download and load the base model) is monkeypatched away, so
no weights, network or GPU are needed.
"""

import json

import pytest
import torch
from torch import nn

from vidlu.training.checkpoint_manager import Files, find_checkpoint_dir, get_checkpoint_dirs
from vidlu_irap_gaim.vlm.finetuning.loading import (CLASSIFIER_CLASSES,
                                                    CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID,
                                                    MODEL_STATE_FILE_NAME, find_model_state_path,
                                                    load_base_classifier,
                                                    load_finetuned_classifier,
                                                    make_classifier_from_config)
from vidlu_irap_gaim.vlm.finetuning.model import Gemma4VLClassifier, Qwen3VLClassifier


def make_checkpoint(experiment_dir, name, perf, state=None):
    """Creates a checkpoint directory as `Checkpoint.save` does, but with only what is read here."""
    checkpoint_dir = experiment_dir / name
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / Files.perf[0]).write_text(json.dumps(perf))
    if state is not None:
        torch.save(state, checkpoint_dir / MODEL_STATE_FILE_NAME)
    return checkpoint_dir


@pytest.fixture
def experiment_dir(tmp_path):
    """An experiment with three checkpoints, the best of which is neither the first nor the last."""
    experiment_dir = tmp_path / "experiment"
    make_checkpoint(experiment_dir, "0_1_0.300", 0.3)
    make_checkpoint(experiment_dir, "1_2_0.700", 0.7)
    make_checkpoint(experiment_dir, "2_3_0.500", 0.5)
    (experiment_dir / "wandb_run_id.txt").write_text("not-a-checkpoint")
    return experiment_dir


def test_find_checkpoint_dir(experiment_dir):
    assert find_checkpoint_dir(experiment_dir, "best").name == "1_2_0.700"
    assert find_checkpoint_dir(experiment_dir, "last").name == "2_3_0.500"


def test_get_checkpoint_dirs_ignores_other_entries(experiment_dir):
    assert [p.name for p in get_checkpoint_dirs(experiment_dir)] == [
        "0_1_0.300", "1_2_0.700", "2_3_0.500"]


def test_checkpoint_index_is_compared_numerically(tmp_path):
    """A ten-checkpoint experiment: "10" is the last one, even though "9" sorts after it."""
    experiment_dir = tmp_path / "experiment"
    for index in range(11):
        make_checkpoint(experiment_dir, f"{index}_{index}_0.500", 0.5)
    assert find_checkpoint_dir(experiment_dir, "last").name == "10_10_0.500"


def test_find_checkpoint_dir_without_checkpoints(tmp_path):
    (empty := tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        find_checkpoint_dir(empty)


def test_find_model_state_path_accepts_all_three_path_kinds(experiment_dir):
    checkpoint_dir = make_checkpoint(experiment_dir, "3_4_0.900", 0.9, state={"_config": {}})
    state_path = checkpoint_dir / MODEL_STATE_FILE_NAME
    assert find_model_state_path(experiment_dir, which="best") == state_path
    assert find_model_state_path(checkpoint_dir) == state_path
    assert find_model_state_path(state_path) == state_path


def test_find_model_state_path_reports_a_checkpoint_without_a_model(experiment_dir):
    with pytest.raises(FileNotFoundError, match=MODEL_STATE_FILE_NAME):
        find_model_state_path(experiment_dir, which="best")


def test_config_round_trips_through_the_loader(monkeypatch):
    """`_get_config_dict` records everything `make_classifier_from_config` needs, and nothing else."""
    monkeypatch.setattr(Qwen3VLClassifier, "_load", lambda self: None)
    original = Qwen3VLClassifier(lora_r=64, lora_alpha=16, lora_dropout=0.1,
                                 lora_target_modules=("q_proj", "v_proj"))
    config = original._get_config_dict()
    assert config["classifier_class"] == "Qwen3VLClassifier"

    model = make_classifier_from_config(config)
    assert type(model) is Qwen3VLClassifier
    for name in ("model_id", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
        assert getattr(model, name) == getattr(original, name)


def test_make_classifier_from_config_requires_a_known_class():
    config = dict(model_id="a/b", classifier_class="NoSuchClassifier")
    with pytest.raises(ValueError, match="Unknown classifier class"):
        make_classifier_from_config(config)
    with pytest.raises(ValueError, match="does not record"):
        make_classifier_from_config(dict(model_id="a/b"))
    # A checkpoint predating `classifier_class` is loadable if the class is named explicitly.
    assert type(make_classifier_from_config(dict(model_id="a/b"),
                                            classifier_class=Qwen3VLClassifier)) \
        is Qwen3VLClassifier


def test_a_checkpoint_predating_classifier_class_is_identified_by_its_base_model(monkeypatch):
    """Such checkpoints still load, as long as they did not override their class's default base
    model -- which is the common case, since the model is usually left to the class."""
    monkeypatch.setattr(Qwen3VLClassifier, "_load", lambda self: None)
    monkeypatch.setattr(Gemma4VLClassifier, "_load", lambda self: None)
    for cls in (Qwen3VLClassifier, Gemma4VLClassifier):
        config = {k: v for k, v in cls()._get_config_dict().items() if k != "classifier_class"}
        assert type(make_classifier_from_config(config)) is cls


def test_default_model_ids_identify_the_classes_uniquely():
    """What `CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID` needs in order not to lose a class."""
    assert len(CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID) == len(CLASSIFIER_CLASSES)
    assert all(model_id == cls._DEFAULT_MODEL_ID
               for model_id, cls in CLASSIFIER_CLASSES_BY_DEFAULT_MODEL_ID.items())


def test_all_classifier_classes_are_registered_under_their_own_names():
    assert all(name == cls.__name__ for name, cls in CLASSIFIER_CLASSES.items())


def test_load_finetuned_classifier(tmp_path, monkeypatch):
    """The loaded model comes from the checkpoint's config, on the requested device, ready to run."""
    monkeypatch.setattr(Qwen3VLClassifier, "_load", lambda self: None)
    loaded_states = []
    monkeypatch.setattr(Qwen3VLClassifier, "load_state_dict",
                        lambda self, state_dict, strict=True: loaded_states.append(state_dict))

    config = Qwen3VLClassifier(lora_r=8)._get_config_dict()
    experiment_dir = tmp_path / "experiment"
    make_checkpoint(experiment_dir, "0_1_0.300", 0.3,
                    state={"_config": config, "_adapter_state": {}})
    best = make_checkpoint(experiment_dir, "1_2_0.700", 0.7,
                           state={"_config": {**config, "lora_r": 64}, "_adapter_state": {}})

    model = load_finetuned_classifier(experiment_dir, device="cpu")

    assert model.lora_r == 64, "the best checkpoint should be selected by default"
    assert model._device == "cpu"
    assert not model.use_gradient_checkpointing
    assert not model.training
    assert loaded_states == [torch.load(best / MODEL_STATE_FILE_NAME, weights_only=False)]


def test_load_finetuned_classifier_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(Qwen3VLClassifier, "_load", lambda self: None)
    monkeypatch.setattr(Qwen3VLClassifier, "load_state_dict",
                        lambda self, state_dict, strict=True: None)
    config = Qwen3VLClassifier()._get_config_dict()
    checkpoint_dir = make_checkpoint(tmp_path / "experiment", "0_1_0.300", 0.3,
                                     state={"_config": config})

    model = load_finetuned_classifier(checkpoint_dir, device="cpu", load_in_4bit=False,
                                      enable_thinking=True)

    assert model.load_in_4bit is False
    assert model.enable_thinking is True


def test_load_finetuned_classifier_rejects_a_foreign_checkpoint(tmp_path):
    checkpoint_dir = make_checkpoint(tmp_path / "experiment", "0_1_0.300", 0.3,
                                     state={"some_layer.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="_config"):
        load_finetuned_classifier(checkpoint_dir, device="cpu")


def test_load_base_classifier(monkeypatch):
    """The pretrained model, with no adapter and nothing to train."""
    monkeypatch.setattr(Qwen3VLClassifier, "_load", lambda self: None)

    model = load_base_classifier(device="cpu")

    assert type(model) is Qwen3VLClassifier
    assert model.model_id == Qwen3VLClassifier._DEFAULT_MODEL_ID
    assert not model.use_lora
    assert not model.use_gradient_checkpointing
    assert not model.training
    assert model._device == "cpu"


def test_load_base_classifier_takes_a_model_and_a_class(monkeypatch):
    monkeypatch.setattr(Gemma4VLClassifier, "_load", lambda self: None)

    model = load_base_classifier("some/fork-of-gemma", classifier_class=Gemma4VLClassifier,
                                 device="cpu", load_in_4bit=False)

    assert type(model) is Gemma4VLClassifier
    assert model.model_id == "some/fork-of-gemma"
    assert model.load_in_4bit is False


def test_load_base_classifier_infers_the_class_from_the_model_id(monkeypatch):
    """A base model records no class of its own, but its id names one whenever it is
    a class's default -- and the classes differ in chat template and processor, so
    defaulting to Qwen would silently malform Gemma's prompts."""
    monkeypatch.setattr(Gemma4VLClassifier, "_load", lambda self: None)

    model = load_base_classifier(Gemma4VLClassifier._DEFAULT_MODEL_ID, device="cpu")

    assert type(model) is Gemma4VLClassifier


def test_load_base_classifier_rejects_an_unrecognized_model_id(monkeypatch):
    """Ambiguity is an error, not a guess."""
    with pytest.raises(ValueError, match="classifier-class"):
        load_base_classifier("some/fork-of-gemma", device="cpu")


def test_loading_without_lora_leaves_the_pretrained_model_unwrapped(monkeypatch):
    """The point of `use_lora=False`: the pretrained module itself, not one wrapped in an adapter.

    This exercises the real `_load`, with only the Hugging Face calls stubbed out, so it would
    catch the adapter creeping back in.
    """
    import transformers

    pretrained = nn.Linear(2, 2)
    monkeypatch.setattr(Qwen3VLClassifier, "make_load_kwargs", lambda self, **kwargs: {})
    monkeypatch.setattr(Qwen3VLClassifier, "_build_hf_model",
                        lambda self, load_kwargs: pretrained)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda *args, **kwargs: "processor")

    model = load_base_classifier(device="cpu")

    assert model._model is pretrained
    assert model.processor == "processor"
