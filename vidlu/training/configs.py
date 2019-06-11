from functools import partial

import torch
from torch.optim import SGD
import numpy as np

from vidlu.utils.func import default_args, params, Empty
from vidlu.utils.misc import fuse
from vidlu.utils.collections import NameDict
from .adversarial.attacks import PGDAttack
from .lr_schedulers import ScalableMultiStepLR, ScalableLambdaLR


# Training/evaluation steps ########################################################################

def supervised_eval_step(trainer, batch):
    trainer.model.eval()
    with torch.no_grad():
        x, y = trainer.prepare_batch(batch)
        prediction, outputs = trainer.get_outputs(x)
        loss = trainer.loss(prediction, y)
        return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item())


def supervised_train_step(trainer, batch):
    trainer.model.train()
    trainer.optimizer.zero_grad()
    x, y = trainer.prepare_batch(batch)
    prediction, outputs = trainer.get_outputs(x)
    loss = trainer.loss(prediction, y)
    loss.backward()
    trainer.optimizer.step()
    return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item())


def autoencoder_train_step(trainer, batch):
    trainer.model.train()
    trainer.optimizer.zero_grad()
    x = trainer.prepare_batch(batch)[0]
    x_r = trainer.model(x)
    loss = trainer.loss(x_r, x)
    loss.backward()
    trainer.optimizer.step()
    return NameDict(reconstruction=x_r, input=x, loss=loss.item())


def adversarial_eval_step(trainer, batch):
    trainer.model.eval()
    with torch.no_grad():
        x, y = trainer.prepare_batch(batch)

        prediction, outputs = trainer.get_outputs(x)
        loss = trainer.loss(prediction, y)

        x_adv = x  # self.attack.perturb(x, y)
        prediction_adv, outputs_adv = prediction, outputs  # self.get_outputs(x_adv)
        loss_adv = loss * 0  # self.loss(prediction_adv, y)
        return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item(),
                        x_adv=x_adv, prediction_adv=prediction_adv, outputs_adv=outputs_adv,
                        loss_adv=loss_adv.item())


class AdversarialTrainStep:
    def __init__(self, adversarial_only=False):
        self.adversarial_only = adversarial_only

    def __call__(self, trainer, batch):
        trainer.model.train()
        x, y = trainer.prepare_batch(batch)

        trainer.optimizer.zero_grad()
        prediction, outputs = trainer.get_outputs(x)
        loss = trainer.loss(prediction, y)
        if not self.adversarial_only:
            loss.backward()
            trainer.optimizer.step()

        x_adv = trainer.attack.perturb(x, y)
        trainer.optimizer.zero_grad()
        prediction_adv, outputs_adv = trainer.get_outputs(x_adv)
        loss_adv = trainer.loss(prediction_adv, y)
        loss_adv.backward()
        trainer.optimizer.step()

        return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item(), x=x,
                        x_adv=x_adv, prediction_adv=prediction_adv, outputs_adv=outputs_adv,
                        loss_adv=loss_adv.item())


def adversarial_train_step(trainer, batch):
    trainer.model.train()
    x, y = trainer.prepare_batch(batch)

    trainer.optimizer.zero_grad()
    prediction, outputs = trainer.get_outputs(x)
    loss = trainer.loss(prediction, y)
    loss.backward()
    trainer.optimizer.step()

    x_adv = trainer.attack.perturb(x, y)
    trainer.optimizer.zero_grad()
    prediction_adv, outputs_adv = trainer.get_outputs(x_adv)
    loss_adv = trainer.loss(prediction_adv, y)
    loss_adv.backward()
    trainer.optimizer.step()

    return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item(), x=x,
                    x_adv=x_adv, prediction_adv=prediction_adv, outputs_adv=outputs_adv,
                    loss_adv=loss_adv.item())


def classification_get_outputs(tr, *x):
    log_probs = tr.model(*x)
    return log_probs, NameDict(prediction=log_probs, log_probs=log_probs, probs=log_probs.exp(),
                               hard_prediction=log_probs.argmax(1))


madry_cifar10_attack = partial(PGDAttack,
                               eps=8 / 255,
                               step_size=2 / 255,
                               grad_preprocessing='sign',
                               step_count=10,  # TODO: change
                               clip_bounds=(0, 1))

# Hyperparameter sets ##############################################################################


supervised = dict(
    eval_step=supervised_eval_step,
    train_step=supervised_train_step)

adversarial = dict(
    eval_step=adversarial_eval_step,  # TODO
    train_step=adversarial_train_step)

classification = dict(
    **supervised,
    get_outputs=classification_get_outputs)

autoencoder = dict(
    train_step=autoencoder_train_step,
    eval_step=supervised_eval_step)

resnet_cifar = dict(  # as in www.arxiv.org/abs/1603.05027
    **classification,
    optimizer_f=partial(SGD, lr=1e-1, momentum=0.9, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128)

wrn_cifar = fuse(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    dict(weight_decay=5e-4),
    overridable=['weight_decay'])

densenet_cifar = dict(  # as in www.arxiv.org/abs/1608.06993
    **classification,
    weight_decay=1e-4,
    optimizer_f=partial(SGD, lr=1e-1, momentum=0.9, weight_decay=Empty, nesterov=True),
    epoch_count=100,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1),
    batch_size=64)

small_image_classifier = dict(  # as in www.arxiv.org/abs/1603.05027
    **classification,
    weight_decay=0,
    optimizer_f=partial(SGD, lr=1e-2, momentum=0.9),
    epoch_count=50,
    batch_size=64)

ladder_densenet = dict(  # custom
    **classification,
    weight_decay=1e-4,
    optimizer_f=partial(SGD, lr=5e-4, momentum=0.9, weight_decay=Empty),
    lr_scheduler_f=partial(ScalableLambdaLR, lr_lambda=lambda p: (1 - p) ** 1.5),
    epoch_count=40,
    batch_size=4)
