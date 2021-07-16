from functools import partial

import torch
from torch import nn

from vidlu.training.trainers import Evaluator
from vidlu.training.trainers import Trainer
import vidlu.training.steps as vts
import vidlu.configs.training as vct
from vidlu.modules.losses import NLLLossWithLogits
from vidlu.modules.components import ClassificationHead
from vidlu.models import ResNetV2, resnet_v2_backbone
from vidlu.training.robustness.attacks import GradientSignAttack


def get_a_model():
    model = ResNetV2(backbone_f=partial(resnet_v2_backbone, depth=18, small_input=True),
                     head_f=partial(ClassificationHead, 2))
    model(torch.empty(1, 3, 32, 32))
    return model


class TestTrainers:
    def test_evaluator_all_args(self):
        """ This should not fail because all attributes are assigned values.
        """
        Evaluator(model=nn.Linear(5, 3), loss=NLLLossWithLogits(),
                  eval_step=vts.supervised_eval_step)

    def test_trainer_init(self):
        Trainer(**vct.TrainerConfig(
            vct.resnet_cifar, model=get_a_model(),
            loss=partial(NLLLossWithLogits(), ignore_index=-1)).normalized())

    def test_adversarial_trainer_init(self):
        Trainer(**vct.TrainerConfig(
            vct.resnet_cifar, vct.adversarial, model=get_a_model(),
            attack_f=GradientSignAttack,
            loss=partial(NLLLossWithLogits(), ignore_index=-1)).normalized())
