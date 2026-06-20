"""
VLM fine-tuning components for Vidlu integration.

This module provides VLM fine-tuning with LoRA (Qwen3-VL and Gemma 4),
designed to integrate with Vidlu's run.py training infrastructure.

The response scheme controls both how ground-truth responses are formatted
during training and how the model's generated text is parsed during evaluation.
Select it via the ``response_scheme_name`` argument of ``make_vlm_bih_data``.

Available scheme names (see RESPONSE_SCHEME_REGISTRY):
    "standard"        - numbered lines, full value text (default)
    "sparse_standard" - numbered lines, only non-default attributes
    "indexed"        - numbered lines, integer value indices
    "sparse_indexed" - numbered lines, only non-default attrs, integer indices
    "json"           - JSON object with attribute name keys

Prompt detail levels (``detail_level`` argument of ``make_vlm_bih_data``):
    "attr_desc_vals" - attribute description + valid values + default (default)
    "attr_vals"      - attribute name + valid values (no descriptions)
    "attr"           - attribute names only (values learned from data)
    "none"           - empty preamble, no attribute list

Usage examples (one per response scheme)::

    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data()" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='json')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='indexed')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='sparse_standard')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='sparse_indexed')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
 
Usage examples (one per prompt detail level)::

    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='standard', detail_level='attr_desc_vals')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='standard', detail_level='attr_vals')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='standard', detail_level='attr')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
    CUDA_VISIBLE_DEVICES=1 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_vlm_bih_data(response_scheme='standard', detail_level='none')" "standardize" "irap_gaim.Qwen3VLClassifier,lora_r=64" "irap_gaim.vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"

Gemma 4 26B-A4B (bf16 + naive MP on 4× A6000, see
.devdocs/gemma4_26b_a4b_finetuning_resources.md).  Uses
``gemma4_vlm_finetune_trainer`` (eval_batch_size=1, lr=1e-4, wd=1e-3)::

    IRAP_HOME=~/projects/irap_home CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run.py train "irap_gaim.make_vlm_bih_data()" "standardize" "irap_gaim.Gemma4VLClassifier,lora_r=64" "irap_gaim.gemma4_vlm_finetune_trainer" --metrics "irap_gaim.get_irap_metrics()"
"""

from .model import Qwen3VLClassifier, Gemma4VLClassifier
from .dataset import VLMIrapDataset, make_vlm_bih_data, make_vlm_vietnam_data
from .steps import VLMTrainStep, VLMEvalStep
from .predictor import FineTunedVLMPredictor

__all__ = [
    "Qwen3VLClassifier",
    "Gemma4VLClassifier",
    "VLMIrapDataset",
    "make_vlm_bih_data",
    "make_vlm_vietnam_data",
    "VLMTrainStep",
    "VLMEvalStep",
    "FineTunedVLMPredictor",
]
