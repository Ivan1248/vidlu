try:
    import irap_data as _irap_data  # noqa: F401
except ModuleNotFoundError as _e:
    if _e.name != "irap_data":
        raise
    raise ModuleNotFoundError(
        "vidlu_irap_gaim needs the irap-data package from irap-tools"
        " (https://github.com/Ivan1248/irap-tools/tree/main/packages/irap_data). Install it with"
        " `uv pip install -e <irap-tools>/packages/irap_data[torch]`, or see requirements.txt."
    ) from _e

# Data factories (re-exported so they resolve in factory expressions,
# e.g. "irap_gaim.get_irap_metrics(irap_gaim.make_vietnam_data()['train'])")
from irap_data import (
    make_bh_data,
    make_irap_data,
    make_irap_data_by_name,
    make_vietnam_data,
)

# Losses & Metrics
from .losses import MultiAttributeCrossEntropyLoss, multi_attribute_cross_entropy
from vidlu.metrics import MultiAttributeClassificationMetrics, OutputKind
from .metrics import (
    IRAP_ATTRIBUTE_METRIC_NAMES,
    IRAP_MAIN_METRIC,
    IRAP_CLASS_SUPPORT_THRESHOLDS,
    get_irap_attribute_metrics,
    get_irap_metrics,
    irap_metric_names,
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

# Sequential enhancement
from .seq import (
    GeneralLSTMModel,
    IdentityEncoder,
    LabelEmbeddingEncoder,
    export_feats,
    extract_features,
    make_seq_enh_data,
    train_seq_enh,
)
from .training import (
    DynamicBalancedRecallWeights,
    EncoderOptimizerMaker,
    FreezeThenFinetune,
    MultiAttributePseudoLabelStep,
    MultiScaleSupervisedStep,
    make_pseudo_labeled_data,
    make_semisup_data,
    multi_attribute_kl_div_ll,
)

# Training
# Trainer configs are re-exported via .training.configs.__all__ so new configs
# don't need to be added here manually.
from .training.configs import *

# VLM fine-tuning
from .vlm.finetuning import (
    Gemma4VLClassifier,
    Qwen3VLClassifier,
    Qwen35Classifier,
    VLMClassifierPredictor,
    VLMEvalStep,
    VLMIrapDataset,
    VLMTrainStep,
    load_base_classifier,
    load_finetuned_classifier,
    make_vlm_bh_data,
    make_vlm_vietnam_data,
)

# Former names of the iRAP-BH factories. Experiment paths contain the data factory expression,
# so these keep the run.py commands of earlier runs valid for resuming and evaluation.
make_bih_data = make_bh_data
make_vlm_bih_data = make_vlm_bh_data
