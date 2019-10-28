from collections import Mapping
from functools import partial
from dataclasses import dataclass

import torch
from torch import optim

from vidlu.data import Record
from vidlu.modules import get_submodule
from vidlu.transforms import jitter
from vidlu.utils.collections import NameDict
from vidlu.utils.func import default_args, params, Empty, Missing
from vidlu.utils.misc import fuse, dict_difference
from .adversarial import attacks
from .lr_schedulers import ScalableMultiStepLR, ScalableLambdaLR, CosineLR
import torch.nn.functional as F
import vidlu.training.steps as ts
import vidlu.training.extensions as te


# Extend output

def classification_extend_output(output):
    if not isinstance(output, torch.Tensor):
        raise ValueError("The output must ba a `torch.Tensor`.")
    logits = output
    return logits, Record(output=logits, log_probs_=lambda: logits.log_softmax(1),
                          probs_=lambda r: r.log_probs.exp(),
                          hard_prediction_=lambda: logits.argmax(1))


# Optimizer makers

@dataclass
class FineTuningOptimizerMaker:
    finetuning: dict

    def __call__(self, trainer, optimizer_f, **kwargs):
        groups = {
            k: {f'{k}.{k_}': p for k_, p in get_submodule(trainer.model, k).named_parameters()}
            for k in self.finetuning}
        remaining = dict(trainer.model.named_parameters())
        for g in groups.values():
            remaining = dict_difference(remaining, g)
        lr = params(optimizer_f).lr
        params_ = ([dict(params=groups[k].values(), lr=f * lr) for k, f in self.finetuning.items()]
                   + [dict(params=remaining.values())])
        return optimizer_f(params_, weight_decay=trainer.weight_decay, **kwargs)


# Adversarial attacks

madry_cifar10_attack = partial(attacks.PGDAttack,
                               eps=8 / 255,
                               step_size=2 / 255,
                               grad_preprocessing='sign',
                               step_count=10,  # TODO: change
                               clip_bounds=(0, 1))

virtual_pgd_cifar10_attack = partial(madry_cifar10_attack,
                                     get_prediction='hard')

vat_pgd_cifar10_attack = partial(madry_cifar10_attack,
                                 get_prediction='soft',
                                 loss=partial(F.kl_div, reduction='batchmean'))

mnistnet_tent_eval_attack = partial(attacks.PGDAttack,
                                    eps=0.3,
                                    step_size=0.1,
                                    grad_preprocessing='sign',
                                    step_count=40,  # TODO: change
                                    clip_bounds=(0, 1),
                                    stop_on_success=True)


# Trainer configs ##################################################################################

class TrainerConfig(NameDict):
    def __init__(self, *args, **kwargs):
        ext_args = []  # extension factories are concatenated in order of appearance
        all_kwargs = {}  # other kwargs are updated with potential overriding
        for x in args:
            if isinstance(x, TrainerConfig):
                d = dict(**x)
                ext_args.extend(d.pop('extension_fs', ()))
                all_kwargs.update(d)
            elif issubclass(x.func if isinstance(x, partial) else x, te.TrainerExtension):
                ext_args.append(x)
            else:
                raise ValueError(f"Invalid argument type: {type(x).__name__}.")
        ext = tuple(kwargs.pop('extension_fs', ())) + tuple(ext_args)
        all_kwargs.update(kwargs)
        super().__init__(**all_kwargs, extension_fs=ext)

    def with_bound_extension_args(self):
        arg_name_to_ext = dict()
        ext = []
        for e in self.extension_fs:
            names = tuple(params(e).keys())
            values = [self.pop(name, Missing) for name in names]
            args = {k: v for k, v in zip(names, values) if v is not Missing}
            ext.append(partial(e, **args) if len(args) > 0 else e)
            for name in names:
                if name in arg_name_to_ext:
                    raise RuntimeError(f'Multiple extension factories have a parameter "{name}".')
                arg_name_to_ext[name] = e
        return TrainerConfig(**{**self, 'extension_fs': ext})


def to_trainer_args(*args, **kwargs):
    return TrainerConfig(*args, **kwargs).with_bound_extension_args()


supervised = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.supervised_train_step)

adversarial = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.adversarial_eval_step,
    train_step=ts.AdversarialTrainStep())

adversarial_free = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.adversarial_eval_step,
    train_step=ts.AdversarialTrainMultiStep())

# supervised_vat = TrainerConfig(
#     adversarial,
#     train_step=AdversarialCombinedLossTrainStep(use_attack_loss=True, clean_weight=1, adv_weight=1),
#     attack_f=attacks)

semisupervised_vat = TrainerConfig(
    partial(te.SemiSupervisedVAT, attack_f=attacks.VATAttack),
    eval_step=ts.semisupervised_vat_eval_step,
    train_step=ts.SemisupervisedVATTrainStep()
)

classification = TrainerConfig(
    supervised,
    extend_output=classification_extend_output)

autoencoder = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.autoencoder_train_step)

resnet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128,
    jitter=jitter.CifarPadRandomCropHFlip())

resnet_cifar_cosine = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-4))

resnet_cifar_single_classifier_cv = TrainerConfig(
    # as in Computer Vision with a Single (Robust) Classiﬁer
    resnet_cifar,
    # overriding:
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=Empty),
    epoch_count=350,
    batch_size=256,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[3 / 7, 5 / 7], gamma=0.1))

wrn_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    resnet_cifar,
    # overriding
    weight_decay=5e-4)

densenet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1608.06993
    classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=Empty, nesterov=True),
    epoch_count=100,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1),
    batch_size=64,
    jitter=jitter.CifarPadRandomCropHFlip())

small_image_classifier = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    weight_decay=0,
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9),
    epoch_count=50,
    batch_size=64,
    jitter=jitter.CifarPadRandomCropHFlip())

ladder_densenet = TrainerConfig(
    classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.SGD, lr=5e-4, momentum=0.9, weight_decay=Empty),
    lr_scheduler_f=partial(ScalableLambdaLR, lr_lambda=lambda p: (1 - p) ** 1.5),
    epoch_count=40,
    batch_size=4,
    optimizer_maker=FineTuningOptimizerMaker({'backbone.backbone': 1 / 5}),
    jitter=jitter.SegRandomHFlip())

swiftnet_cityscapes = TrainerConfig(
    classification,
    weight_decay=1e-4,  # za imagenet 4 puta manje, tj. 16 puta uz množenje r korakom učenja
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=Empty),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=14,
    eval_batch_size=8,
    optimizer_maker=FineTuningOptimizerMaker({'backbone.backbone': 1 / 4}),
    jitter=jitter.SegRandomScaleCropHFlip(shape=(768, 768), max_scale=2, overstepping='half'))

swiftnet_camvid = TrainerConfig(
    swiftnet_cityscapes,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-7),
    epoch_count=600,  # 600
    batch_size=12,
    jitter=jitter.SegRandomScaleCropHFlip(shape=(448, 448), max_scale=2, overstepping='half'))

swiftnet_camvid_scratch = TrainerConfig(
    swiftnet_camvid,
    # overriding:
    epoch_count=900,
    optimizer_maker=None)

semseg_basic = TrainerConfig(
    classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=Empty),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=40,
    batch_size=8,
    eval_batch_size=8,  # max 12?
    optimizer_maker=FineTuningOptimizerMaker({'backbone': 1 / 4}),
    jitter=jitter.SegRandomCropHFlip((768, 768)))

mnistnet = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, nesterov=True, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=50,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128)

mnistnet_tent = TrainerConfig(
    classification,
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=40,
    lr_scheduler_f=None,
    batch_size=100)

## special


wrn_cifar_tent = TrainerConfig(
    wrn_cifar,
    # overriding
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=Empty),
    weight_decay=1e-4)

resnet_cifar_adversarial_esos = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    adversarial,
    attack_f=partial(madry_cifar10_attack, step_count=7),
    eval_attack_f=partial(madry_cifar10_attack, step_count=10, stop_on_success=True))

resnet_cifar_adversarial_multistep_esos = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar_adversarial_esos,
    # overriding
    train_step=ts.AdversarialTrainMultiStep(train_mode=False, update_period=8),
    epoch_count=25)
