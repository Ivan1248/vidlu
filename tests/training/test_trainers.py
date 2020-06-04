import pytest

from functools import partial

import torch
from torch import nn

from vidlu.training.trainers import Evaluator
from vidlu.training.trainers import Trainer
import vidlu.training.steps as ts
import vidlu.training.configs as tc
from vidlu.modules.losses import NLLLossWithLogits
from vidlu.modules.components import ClassificationHead
from vidlu.models import ResNetV2, resnet_v2_backbone
from vidlu.training.adversarial.attacks import GradientSignAttack


def get_a_model():
    model = ResNetV2(backbone_f=partial(resnet_v2_backbone, depth=18),
                     head_f=partial(ClassificationHead, 2))
    model(torch.empty(1, 3, 32, 32))
    return model


class TestTrainers:
    def test_evaluator_no_args(self):
        """ This should fail when AttributeCheckingMeta tests whether all
        attributes have valid values, i.e. are not equal to `Required`.
        """
        with pytest.raises(TypeError):  # missing overwrite=True
            Evaluator(model=nn.Linear(5, 3))

    def test_evaluator_all_args(self):
        """ This should not fail because all attributes are assigned values.
        """
        Evaluator(model=nn.Linear(5, 3), loss=NLLLossWithLogits(),
                  eval_step=ts.supervised_eval_step)

    def test_trainer_init(self):
        Trainer(**tc.to_trainer_args(tc.resnet_cifar, model=get_a_model(),
                                     loss=partial(NLLLossWithLogits(), ignore_index=-1)))

    def test_adversarial_trainer_init(self):
        Trainer(
            **tc.to_trainer_args(tc.resnet_cifar, tc.adversarial, model=get_a_model(),
                                 attack_f=GradientSignAttack,
                                 loss=partial(NLLLossWithLogits(), ignore_index=-1)))
