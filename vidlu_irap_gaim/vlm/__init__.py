"""
VLM (Vision-Language Model) integration for zero-shot road attribute classification.
"""

from .models import (
    VLMPredictionResult,
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
    predictions_to_output_tuple,
    predictions_to_json_serializable,
)

__all__ = [
    # Base
    "VLMPredictionResult",
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
    "predictions_to_output_tuple",
    "predictions_to_json_serializable",
]
