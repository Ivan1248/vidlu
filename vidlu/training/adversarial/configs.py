from functools import partial

import torch.nn.functional as F
import torch

import vidlu.optim as vo
import vidlu.modules as vm
import vidlu.modules.inputwise as vmi
from vidlu.modules import losses
from vidlu.training.adversarial import attacks

from vidlu.training.adversarial import perturbation_models

# Adversarial attacks

madry_cifar10_attack = partial(attacks.PGDAttack,
                               eps=8 / 255,
                               step_size=2 / 255,
                               step_count=10,  # TODO: change
                               clip_bounds=(0, 1))

channelwise_pgd_attack = partial(attacks.PerturbationModelAttack,
                                 pert_model_f=partial(vmi.Additive, ()),
                                 pert_model_init=None,
                                 step_size=2 / 255,
                                 step_count=10,  # TODO: change
                                 clip_bounds=(0, 1),
                                 project_params=0.1)

pmodel_attack_1 = partial(attacks.PerturbationModelAttack,
                          pert_model_f=partial(vmi.AlterGamma, (2, 3)),
                          pert_model_init=None,
                          step_size=2 / 255,
                          step_count=10,  # TODO: change
                          clip_bounds=(0, 1),
                          project_params=0.1)

warp_attack = partial(attacks.PerturbationModelAttack,
                      pert_model_f=vmi.Warp,
                      pert_model_init=None,
                      step_size=0.25,
                      step_count=7,  # TODO: change
                      clip_bounds=(0, 1),
                      project_params=1.)  # TODO: semantic segmentation

morsic_tps_warp_attack = partial(attacks.PerturbationModelAttack,
                                 # pert_model_f=partial(vmi.MorsicTPSWarp, grid_shape=(2, 2),
                                 #                      label_padding_mode='zeros'),
                                 pert_model_f=partial(perturbation_models.AlgTohchabTorbiwasc),
                                 # pert_model_init=lambda pmodel: pmodel.theta.uniform_(-.1, .1),
                                 pert_model_init=lambda pmodel: vmi.reset_parameters(pmodel),
                                 step_size=0.01,
                                 step_count=7,
                                 project_params=.3)  # TODO: semantic segmentation

entmin_attack = partial(madry_cifar10_attack,
                        minimize=False,
                        loss=lambda logits, _: losses.entropy(logits))

virtual_pgd_cifar10_attack = partial(madry_cifar10_attack,
                                     get_prediction='hard')

vat_pgd_cifar10_attack = partial(madry_cifar10_attack,
                                 get_prediction='soft',
                                 loss=partial(F.kl_div, reduction='batchmean'))

mnistnet_tent_eval_attack = partial(attacks.PGDAttack,
                                    eps=0.3,
                                    step_size=0.1,
                                    grad_processing='sign',
                                    step_count=40,  # TODO: change
                                    clip_bounds=(0, 1),
                                    stop_on_success=True)
