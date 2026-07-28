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
    ImageSequenceClassifier,
    ResNetEncoder,
    ViTEncoder,
    dinov2_vit_encoder,
    MultiScaleSequenceInference,
)
from .models.pretraining import vistas_params_spec

# Losses & Metrics
from .losses import multi_attribute_cross_entropy, MultiAttributeCrossEntropyLoss
from .metrics import (
    MultiAttributeClassificationMetrics,
    MultiAttributeAccuracy,
    get_irap_metrics,
)

# Training
# Trainer configs come from .training.configs.__all__, so new ones need no change here.
from .training.configs import *
from .training import (
    FreezeThenFinetune,
    MultiScaleSupervisedStep,
    MultiAttributePseudoLabelStep,
    DynamicBalancedRecallWeights,
    add_attr_idx_to_class_occurrence_counts_to_info_lazily,
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
