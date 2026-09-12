import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

# `irap_data` is a separate project living in `irap-data/` of this checkout, so importing
# it needs that directory on `sys.path` – the repo root alone is not enough. `scripts/run.py`
# gets this from `scripts/_context.py`, but entry points that do not go through it (notably
# `python -m vidlu_irap_gaim.tools.<name>`) otherwise fail here with ModuleNotFoundError.
# Installed environments already resolve `irap_data` and are left untouched.
if _importlib_util.find_spec("irap_data") is None:
    _irap_data_project = _Path(__file__).resolve().parent.parent / "irap-data"
    if _irap_data_project.is_dir():
        _sys.path.insert(0, str(_irap_data_project))

# Data factories (re-exported so they resolve in factory expressions,
# e.g. "irap_gaim.get_irap_metrics(irap_gaim.make_vietnam_data()['train'])")
from irap_data import (
    make_irap_data,
    make_bih_data,
    make_vietnam_data,
    make_irap_data_by_name,
)

# Models
from .models import (
    FrameEncoder,
    ImageSequenceClassifier,
    MAEEncoder,
    MultiScaleSequenceInference,
    PixelStats,
    Qwen3VLVisionEncoder,
    ResNetEncoder,
    TimmViTEncoder,
    ViTEncoder,
    dinov2_vit_encoder,
    dinov3_vit_encoder,
    load_spar_trunk_state_dict,
    mae_vit_encoder,
    siglip2_vit_encoder,
)
from .models.pretraining import vistas_params_spec

# Losses & Metrics
from .losses import multi_attribute_cross_entropy, MultiAttributeCrossEntropyLoss
from vidlu.metrics import MultiAttributeClassificationMetrics, OutputKind
from .metrics import (
    IRAP_ATTRIBUTE_METRIC_NAMES,
    IRAP_MAIN_METRIC,
    IRAP_CLASS_SUPPORT_THRESHOLDS,
    get_irap_attribute_metrics,
    get_irap_metrics,
    irap_metric_names,
)

# Training
# Trainer configs are re-exported via .training.configs.__all__ so new configs
# don't need to be added here manually.
from .training.configs import *
from .training import (
    EncoderOptimizerMaker,
    FreezeThenFinetune,
    MultiScaleSupervisedStep,
    MultiAttributePseudoLabelStep,
    DynamicBalancedRecallWeights,
    multi_attribute_kl_div_ll,
    make_semisup_data,
    make_pseudo_labeled_data,
)

# Sequential enhancement
from .seq import export_feats, make_seq_enh_data, GeneralLSTMModel

# VLM fine-tuning
from .vlm.finetuning import (
    Qwen3VLClassifier,
    Gemma4VLClassifier,
    make_vlm_bih_data,
    make_vlm_vietnam_data,
    VLMIrapDataset,
    VLMTrainStep,
    VLMEvalStep,
    FineTunedVLMPredictor,
)
