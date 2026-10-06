"""What `vlm_inference.run_evaluation` records, with a stub predictor, so it runs without a model
or IRAP_HOME."""

import json

import irap_evaluation as ie
import pytest
import torch

from vidlu.data import Dataset
from vidlu_irap_gaim.tools.vlm_inference import run_evaluation, run_evaluation_on_data_entries
from vidlu_irap_gaim.vlm.models.base import BaseVLMPredictor
from vidlu_irap_gaim.vlm.response_scheme import make_response_scheme

ATTR_TO_VALUE_TO_CODE = {"width": {"narrow": 1, "wide": 2},
                         "surface": {"paved": 1, "gravel": 3, "dirt": 4}}
ATTR_TO_VALUE_TO_CLASS_IDX = {a: {v: i for i, v in enumerate(vs)}
                              for a, vs in ATTR_TO_VALUE_TO_CODE.items()}


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text)


class _StubPredictor(BaseVLMPredictor):
    """Responds "wide" and "dirt" to every image."""

    def _load_model(self):
        pass

    def _generate_batch(self, pil_images, prompt, max_response_tokens):
        return [("1: wide\n2: dirt", None, False)] * len(pil_images)

    @property
    def tokenizer(self):
        return _CharacterTokenizer()  # counts the tokens of the responses


@pytest.fixture
def dataset_info(tmp_path):
    attr_meta = {"attribute_to_idx": {a: i for i, a in enumerate(ATTR_TO_VALUE_TO_CODE)},
                 "attribute_value_to_irap_number": ATTR_TO_VALUE_TO_CODE}
    (tmp_path / "attribute_metadata.json").write_text(json.dumps(attr_meta), encoding="utf-8")
    return {"dataset_name": "vietnam", "split": "test", "context_offsets": [0],
            "metadata_dir": str(tmp_path),
            "attr_to_value_to_class_idx": ATTR_TO_VALUE_TO_CLASS_IDX,
            "vlm_response_scheme": make_response_scheme("standard", ATTR_TO_VALUE_TO_CLASS_IDX),
            "vlm_attrs_to_include": list(ATTR_TO_VALUE_TO_CODE)}


def _dataset(info, split="test"):
    examples = [{"image": torch.zeros(3, 8, 8), "segment_id": f"s{i}"} for i in range(3)]
    return Dataset(data=examples, info={**info, "split": split})


def _predictor(**kwargs):
    return _StubPredictor("stub-model", max_response_tokens=64, **kwargs)


_EVALUATION_ARGS = dict(interactive=False, print_prompt=False, batch_size=2)


def test_the_files_record_the_dataset_and_the_predictor(tmp_path, dataset_info):
    """Not the defaults of the arguments that a supplied dataset or predictor makes unused."""
    model_info = ie.ModelInfo(method_name="m", training_splits=("train",),
                              early_stopping_splits=(), seed=3, details={"checkpoint": "c"})
    output_dir = tmp_path / "out"
    result = run_evaluation(_dataset(dataset_info), _predictor(enable_thinking=True,
                                                               temperature=0.5),
                            output_dir, model_info, **_EVALUATION_ARGS)

    assert result.num_samples_completed == 3
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert (summary["dataset"], summary["split"], summary["model_id"]) == \
           ("vietnam", "test", "stub-model")

    predictions = ie.read_predictions(output_dir / "m_seed3.test.predictions.parquet")
    assert predictions.segment_ids == ("s0", "s1", "s2")
    assert predictions.probs["surface"][0].tolist() == [0, 0, 1]
    header_model = predictions.header.model
    assert (header_model.method_name, header_model.seed, header_model.training_splits) == \
           ("m", 3, ("train",))
    assert header_model.details["checkpoint"] == "c" and "commit" in header_model.details
    assert header_model.details["configuration"] | {"model_id": "stub-model",
                                                    "enable_thinking": True,
                                                    "temperature": 0.5} == \
           header_model.details["configuration"]


def test_data_entries_are_evaluated_by_key_prefix(tmp_path, dataset_info):
    data = {"train": _dataset(dataset_info, "train"), "test": _dataset(dataset_info),
            "test_hard": _dataset(dataset_info)}
    model_info = ie.ModelInfo(method_name="m", training_splits=(), early_stopping_splits=())

    results = run_evaluation_on_data_entries(data, "test", _predictor(), tmp_path, model_info,
                                             **_EVALUATION_ARGS)

    assert list(results) == ["test", "test_hard"]
    assert (tmp_path / "test_hard" / "m.test.predictions.parquet").exists()
    with pytest.raises(ValueError):
        run_evaluation_on_data_entries(data, "val", _predictor(), tmp_path, model_info)
