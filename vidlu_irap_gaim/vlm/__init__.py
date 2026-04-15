"""
VLM (Vision-Language Model) integration for zero-shot road attribute classification.
"""

from .base import (
    VLMPredictionResult,
    BaseVLMPredictor,
    extract_center_frame,
)
from .qwen3_vl import Qwen3VLPredictor
from .qwen3_vl_vllm import Qwen3VLvLLMPredictor
from .gemma4_vl import Gemma4VLPredictor
from .gemma4_vl_vllm import Gemma4VLvLLMPredictor
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
    predictions_to_output_tuple,
    predictions_to_json_serializable,
)

__all__ = [
    # Base
    "VLMPredictionResult",
    "BaseVLMPredictor",
    "extract_center_frame",
    # Predictors
    "Qwen3VLPredictor",
    "Qwen3VLvLLMPredictor",
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
    # Response parsing (low-level utilities)
    "AttributePrediction",
    "parse_vlm_response",
    "predictions_to_output_tuple",
    "predictions_to_json_serializable",
]
