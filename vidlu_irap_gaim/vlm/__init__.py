"""
VLM (Vision-Language Model) integration for zero-shot road attribute classification.
"""

from .models import (
    VLMPredictionResult,
    VLMSessionResult,
    attribute_sessions,
    BaseVLMPredictor,
    Qwen3VLPredictor,
    Qwen3VLvLLMPredictor,
    Qwen36VLvLLMPredictor,
    Gemma4VLPredictor,
    Gemma4VLvLLMPredictor,
)
from .prompts import PromptBuilder, DetailLevel
from .response_scheme import (
    ResponseScheme,
    StandardResponseScheme,
    JsonResponseScheme,
    SparseStandardResponseScheme,
    IndexedResponseScheme,
    SparseIndexedResponseScheme,
    registry,
    make_response_scheme,
)
from .response_parser import (
    AttributePrediction,
    parse_vlm_response,
)
from .predictions import (
    attribute_predictions_to_one_hot_outputs,
    predictions_to_json_serializable,
)

__all__ = [
    # Base
    "VLMPredictionResult",
    "VLMSessionResult",
    "attribute_sessions",
    "BaseVLMPredictor",
    # Predictors
    "Qwen3VLPredictor",
    "Qwen3VLvLLMPredictor",
    "Qwen36VLvLLMPredictor",
    "Gemma4VLPredictor",
    "Gemma4VLvLLMPredictor",
    # Prompts
    "PromptBuilder",
    "DetailLevel",
    # Response schemes
    "ResponseScheme",
    "StandardResponseScheme",
    "JsonResponseScheme",
    "SparseStandardResponseScheme",
    "IndexedResponseScheme",
    "SparseIndexedResponseScheme",
    "registry",
    "make_response_scheme",
    # Response parsing
    "AttributePrediction",
    "parse_vlm_response",
    # Predictions (tensor/output conversion)
    "attribute_predictions_to_one_hot_outputs",
    "predictions_to_json_serializable",
]
