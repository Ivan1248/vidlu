"""
Helper functions for configuring IRAP GAIM experiments with attrs_to_include.
"""

from functools import partial
from typing import Sequence

from vidlu.configs.training import TrainerConfig
from vidlu.training.steps import SupervisedStep
import torch

from vidlu_irap_gaim.losses import MultiAttributeCrossEntropyLoss
from .dynamic_weights import DynamicBalancedRecallWeights
from .extensions import FreezeThenFinetune
from .jitter import make_sequence_color_jitter
from irap_data.attrs import get_attrs_to_include


def make_irap_trainer_with_attrs(
    attrs_to_include: Sequence[str] | None = None,
    frozen_epochs: int = 2,
    finetune_epochs: int = 8,
    batch_size: int = 12,
    eval_batch_size: int = 32,
    frozen_lr: float = 5e-5,
    finetune_lr: float = 1e-5,
    frozen_weight_decay: float = 1e-3,
    finetune_weight_decay: float = 1e-3,
    frozen_lr_decay: float = 0.8,
    finetune_lr_decay: float = 0.88,
    jitter=make_sequence_color_jitter(),
) -> TrainerConfig:
    """
    Creates an IRAP trainer config with attribute subset filtering.

    This helper configures the trainer to use `attrs_to_include` for loss computation,
    dynamic weighting, and metrics. If `attrs_to_include` is None, uses the canonical
    paper subset from `get_attrs_to_include()`.

    Args:
        attrs_to_include: Optional sequence of attribute names to include. If None,
            uses the canonical paper subset.
        frozen_epochs: Number of epochs in frozen phase.
        finetune_epochs: Number of epochs in finetune phase.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        frozen_lr: Learning rate for frozen phase.
        finetune_lr: Learning rate for finetune phase.
        frozen_weight_decay: Weight decay for frozen phase.
        finetune_weight_decay: Weight decay for finetune phase.
        frozen_lr_decay: Per-epoch multiplicative LR decay factor for the frozen phase.
        finetune_lr_decay: Per-epoch multiplicative LR decay factor for the finetune phase.

    Returns:
        A TrainerConfig configured for IRAP local recognition with attribute filtering.

    Note:
        The returned config will need `attrs_idx` to be set on the metrics and loss
        at runtime based on the dataset's attribute order. This is typically done via
        a factory expression that calls `map_attr_names_to_indices`.
    """
    if attrs_to_include is None:
        attrs_to_include = get_attrs_to_include()

    return TrainerConfig(
        eval_step=SupervisedStep(eval=True, amp=True),
        train_step=SupervisedStep(amp=True),
        loss=MultiAttributeCrossEntropyLoss(),
        optimizer_f=partial(torch.optim.Adam, lr=frozen_lr, weight_decay=frozen_weight_decay),
        epoch_count=frozen_epochs + finetune_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        jitter=jitter,
        extension_fs=[
            partial(
                FreezeThenFinetune,
                num_frozen_epochs=frozen_epochs,
                frozen_lr=frozen_lr,
                finetune_lr=finetune_lr,
                frozen_weight_decay=frozen_weight_decay,
                finetune_weight_decay=finetune_weight_decay,
                frozen_lr_total_decay=frozen_lr_decay ** frozen_epochs,
                finetune_lr_total_decay=finetune_lr_decay ** finetune_epochs,
            ),
            DynamicBalancedRecallWeights,  # attrs_idx set at runtime
        ],
    )
