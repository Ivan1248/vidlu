"""Optimizer construction for the iRAP attribute classifier.

The regime an experiment runs in – linear probing, LoRA, or full fine-tuning with
layer-wise LR decay – is a property of the trainer configuration, but it has to take
effect while the model is being wired to the optimizer. :class:`EncoderOptimizerMaker`
is the one object that sees both, so it owns both decisions.
"""

import typing as T

from vidlu.optim import OptimizerMaker

from ..models.encoders import TrainabilityMode


class EncoderOptimizerMaker(OptimizerMaker):
    """Builds an optimizer for an ``ImageSequenceClassifier``, in a chosen training regime.

    Args:
        optimizer_f: The optimizer factory, e.g. ``partial(torch.optim.AdamW)``.
        trainable: Which backbone parameters to train. See
            :data:`~vidlu_irap_gaim.models.encoders.TRAINABILITY_MODES`:
            ``'pool'`` is linear probing, ``'lora'`` adapter tuning, ``'all'`` full
            fine-tuning.
        lr: The learning rate of the backbone's topmost layer.
        weight_decay: Weight decay for the backbone parameters that take it (the encoder
            excludes norms, biases and embeddings when it knows how).
        layer_decay: If given, the per-block multiplicative decay of the learning rate
            towards the input end of the backbone. ``0.65``–``0.75`` is the usual range
            for fine-tuning pretrained ViTs (He et al., *Masked Autoencoders Are Scalable
            Vision Learners*, https://arxiv.org/abs/2111.06377, App. A.1).
        head_lr_multiplier: The attribute heads are randomly initialized, so they are
            trained at ``lr * head_lr_multiplier`` rather than at the backbone's rate.
        head_weight_decay: Weight decay for the heads. Defaults to `weight_decay`.

    Why this is an :class:`~vidlu.optim.utils.OptimizerMaker` subclass: ``Trainer``
    dispatches on that type to pass the *model* instead of ``model.parameters()``, and it
    creates the optimizer before extensions are initialized. So this is both the only
    place with access to the model at configuration time, and the last moment at which
    setting `requires_grad` still affects which parameters the optimizer receives.
    """

    def __init__(self, optimizer_f: T.Callable, *, trainable: TrainabilityMode, lr: float,
                 weight_decay: float = 0., layer_decay: float | None = None,
                 head_lr_multiplier: float = 1., head_weight_decay: float | None = None,
                 **optimizer_kwargs):
        super().__init__(optimizer_f, params=[], ignore_remaining_params=True,
                         **optimizer_kwargs)
        self.trainable = trainable
        self.lr = lr
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.head_lr_multiplier = head_lr_multiplier
        self.head_weight_decay = weight_decay if head_weight_decay is None \
            else head_weight_decay

    def __call__(self, model):
        model.set_encoder_trainable(self.trainable)
        groups = model.frame_encoder.param_groups(
            lr=self.lr, weight_decay=self.weight_decay, layer_decay=self.layer_decay)
        groups.append(dict(params=model.head_parameters(),
                           lr=self.lr * self.head_lr_multiplier,
                           weight_decay=self.head_weight_decay))
        return self.optimizer_f(groups, lr=self.lr, **self.kwargs)

    def __repr__(self):
        return (f"EncoderOptimizerMaker(optimizer_f={self.optimizer_f!r},"
                f" trainable={self.trainable!r}, lr={self.lr}, layer_decay={self.layer_decay},"
                f" head_lr_multiplier={self.head_lr_multiplier}, kwargs={self.kwargs!r})")
