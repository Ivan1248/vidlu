"""
Minimal integration test for Qwen3-VL predictor.
Verifies that the class can be instantiated and the predict method
can be called (with mocked model/processor).
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from vidlu_irap_gaim.vlm.models.qwen3_vl import Qwen3VLPredictor
from vidlu_irap_gaim.vlm.response_parser import AttributePrediction

@pytest.fixture
def mock_transformers():
    with patch("transformers.AutoModelForImageTextToText.from_pretrained") as mock_model_load, \
         patch("transformers.AutoProcessor.from_pretrained") as mock_processor_load:
        
        mock_model = MagicMock()
        mock_processor = MagicMock()
        # The session loop measures each response in tokens; a MagicMock would
        # have no length. Whitespace splitting is enough of a tokenizer here.
        mock_processor.tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: text.split())

        mock_model_load.return_value = mock_model
        mock_processor_load.return_value = mock_processor

        yield mock_model, mock_processor

@pytest.fixture
def sample_metadata():
    return {
        "Area type": {"Urban": 0, "Rural": 1},
        "Number of lanes": {"One": 0, "Two": 1, "Three": 2, "Four or more": 3}
    }

def test_qwen3_predictor_instantiation():
    """Verifies that Qwen3VLPredictor can be instantiated."""
    predictor = Qwen3VLPredictor(model_id="Qwen/Qwen3-VL-8B-Instruct")
    assert predictor.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert predictor._model is None
    assert predictor._processor is None

@patch("vidlu_irap_gaim.vlm.models.qwen3_vl.Qwen3VLPredictor._generate_batch")
def test_qwen3_predictor_predict_basic(mock_generate, mock_transformers, sample_metadata):
    """Verifies the predict method works with mocked generation."""
    mock_model, mock_processor = mock_transformers
    mock_model.device = "cpu"

    # _generate_batch responds to one prompt for a batch of images, returning
    # (response, thinking, is_truncated) per image.
    mock_generate.return_value = [
        ('{"Area type": "Urban", "Number of lanes": "Two"}', None, False)]

    predictor = Qwen3VLPredictor(device="cpu", max_response_tokens=512)

    # Create dummy image
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    result = predictor.predict(
        image,
        sample_metadata,
        attrs_to_include=["Area type", "Number of lanes"]
    )

    assert isinstance(result.predictions, dict)
    assert "Area type" in result.predictions
    assert result.predictions["Area type"].pred_value == "Urban"
    assert result.predictions["Area type"].pred_idx == 0
    assert result.predictions["Number of lanes"].pred_idx == 1
    assert "Urban" in result.raw_response


@patch("vidlu_irap_gaim.vlm.models.qwen3_vl.Qwen3VLPredictor._generate_batch")
def test_one_attribute_per_session_runs_one_session_per_attribute(
        mock_generate, mock_transformers, sample_metadata):
    """The per-attribute arm: each attribute gets its own prompt and its own
    exchange, and each session's response is recorded separately – a merged blob
    could not say which response belonged to which question."""
    mock_model, mock_processor = mock_transformers
    mock_model.device = "cpu"
    mock_generate.side_effect = [[("1: Urban", None, False)],
                                 [("1: Two", None, False)]]

    predictor = Qwen3VLPredictor(device="cpu", attrs_per_session=1,
                                 max_response_tokens=512)
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    result = predictor.predict(image, sample_metadata,
                               attrs_to_include=["Area type", "Number of lanes"])

    assert mock_generate.call_count == 2
    assert [s.attrs for s in result.sessions] == [["Area type"], ["Number of lanes"]]
    assert [s.response for s in result.sessions] == ["1: Urban", "1: Two"]
    assert result.predictions["Area type"].pred_idx == 0
    assert result.predictions["Number of lanes"].pred_idx == 1
    # Each prompt asks about exactly one attribute.
    assert "Number of lanes" not in result.sessions[0].prompt
    assert "Area type" not in result.sessions[1].prompt
