"""Round trips of the prediction files written by `vidlu_irap_gaim.prediction_files`.

The dataset is a fake `info` with a two-attribute `attribute_metadata.json`, so these run
without IRAP_HOME.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import irap_evaluation as ie
from vidlu_irap_gaim.prediction_files import (make_prediction_file_spec, write_output_predictions,
                                              write_parsed_predictions)
from vidlu_irap_gaim.tools.inference import _OutputCollector
from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

ATTR_TO_VALUE_TO_CODE = {"width": {"narrow": 1, "wide": 2},
                         "surface": {"paved": 1, "gravel": 3, "dirt": 4}}
ATTR_TO_VALUE_TO_CLASS_IDX = {a: {v: i for i, v in enumerate(vs)}
                              for a, vs in ATTR_TO_VALUE_TO_CODE.items()}


@pytest.fixture
def dataset_info(tmp_path):
    attr_meta = {"attribute_to_idx": {a: i for i, a in enumerate(ATTR_TO_VALUE_TO_CODE)},
                 "attribute_value_to_irap_number": ATTR_TO_VALUE_TO_CODE}
    (tmp_path / "attribute_metadata.json").write_text(json.dumps(attr_meta), encoding="utf-8")
    return {"dataset_name": "bh", "split": "test", "context_offsets": [0],
            "metadata_dir": str(tmp_path)}


def _prediction(attr, value):
    return AttributePrediction(attr, value, ATTR_TO_VALUE_TO_CLASS_IDX[attr].get(value, -1))


def test_spec_needs_a_release_split(dataset_info):
    with pytest.raises(ValueError):
        make_prediction_file_spec({**dataset_info, "dataset_name": None}, "m")


def test_output_predictions_round_trip(tmp_path, dataset_info):
    spec = make_prediction_file_spec(dataset_info, "m", 1, {"checkpoint": "c"})
    probs = [np.array([[0.25, 0.75], [1.0, 0.0]]), np.array([[0.5, 0.25, 0.25], [0, 0, 1.0]])]
    path = tmp_path / "m.predictions.parquet"
    write_output_predictions(path, spec, ["s0", "s1"], probs, "probs")

    predictions = ie.read_predictions(path)
    assert predictions.segment_ids == ("s0", "s1")
    assert predictions.attribute_to_irap_codes == {"width": (1, 2), "surface": (1, 3, 4)}
    np.testing.assert_allclose(predictions.probs["surface"], probs[1])
    assert predictions.header.method.seed == 1


def test_output_predictions_refuse_hard_outputs(tmp_path, dataset_info):
    spec = make_prediction_file_spec(dataset_info, "m")
    with pytest.raises(ValueError):
        write_output_predictions(tmp_path / "m.parquet", spec, ["s0"],
                                 [np.eye(2)[:1], np.eye(3)[:1]], "hard")


def test_parsed_predictions_mark_unusable_responses_invalid(tmp_path, dataset_info):
    spec = make_prediction_file_spec(dataset_info, "m")
    segment_to_predictions = {
        "s0": {"width": _prediction("width", "wide"), "surface": _prediction("surface", "?")},
        "s1": {},  # no response
        "s2": {"surface": _prediction("surface", "dirt")},
    }
    path = tmp_path / "m.predictions.parquet"
    write_parsed_predictions(path, spec, segment_to_predictions, ["surface"])

    predictions = ie.read_predictions(path)
    assert predictions.segment_ids == ("s0", "s1", "s2")
    assert predictions.attribute_to_irap_codes == {"surface": (1, 3, 4)}
    assert predictions.is_valid["surface"].tolist() == [False, False, True]
    assert predictions.probs["surface"][2].tolist() == [0, 0, 1]


def test_output_collector_concatenates_batches():
    collector = _OutputCollector()
    for segment_ids, outputs in [(["s0", "s1"], (torch.ones(2, 2), torch.zeros(2, 3))),
                                 (["s2"], (torch.full((1, 2), 2.), torch.ones(1, 3)))]:
        collector.on_iter_completed(SimpleNamespace(batch={"segment_id": segment_ids},
                                                    result=SimpleNamespace(out=outputs)))
    width, surface = collector.get_outputs()
    assert collector.segment_ids == ["s0", "s1", "s2"]
    assert width.tolist() == [[1, 1], [1, 1], [2, 2]]
    assert surface.shape == (3, 3)
