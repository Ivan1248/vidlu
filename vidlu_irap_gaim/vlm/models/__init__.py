"""
VLM model implementations: base classes, model-family predictors, and utilities.
"""

from .base import VLMPredictionResult, BaseVLMPredictor
from .base_hf import BaseHFPredictor
from .base_vllm import BaseVLLMPredictor
from .gemma4_vl import Gemma4VLPredictor
from .gemma4_vl_vllm import Gemma4VLvLLMPredictor
from .gemma_utils import build_gemma_chat_messages
from .qwen3_vl import Qwen3VLPredictor
from .qwen3_vl_vllm import Qwen3VLvLLMPredictor
from .qwen36_vl_vllm import Qwen36VLvLLMPredictor
from .qwen_utils import build_qwen_chat_messages
from .thinking import strip_thinking

__all__ = [
    "VLMPredictionResult",
    "BaseVLMPredictor",
    "BaseHFPredictor",
    "BaseVLLMPredictor",
    "Gemma4VLPredictor",
    "Gemma4VLvLLMPredictor",
    "build_gemma_chat_messages",
    "Qwen3VLPredictor",
    "Qwen3VLvLLMPredictor",
    "Qwen36VLvLLMPredictor",
    "build_qwen_chat_messages",
    "strip_thinking",
]
