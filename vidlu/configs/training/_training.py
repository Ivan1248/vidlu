from torch import optim

from vidlu.transforms import jitter
from vidlu.optim.lr_schedulers import ScalableMultiStepLR, ScalableLR, CosineLR
import vidlu.training.steps as ts
import vidlu.training.extensions as te

import vidlu.data as vd
import vidlu.data.utils as vdu
from vidlu.configs.robustness import *
from .trainer_config import TrainerConfig, OptimizerMaker
from .classification import classification_extend_output

# Basic (train_step, eval_step, extend_output)

supervised = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.supervised_train_step,
)

classification = TrainerConfig(supervised, extend_output=classification_extend_output)

# Adversarial training, basic

adversarial = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialTrainStep(),
)

adversarial_combined = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialCombinedLossTrainStep(),
)

# Adversarial training

adversarial_free = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialTrainMultiStep(),
)

# VAT

vat = TrainerConfig(
    adversarial,
    train_step=ts.VATTrainStep(),
    eval_step=ts.AdversarialEvalStep(virtual=True),
    attack_f=attacks.VATAttack,
)

# Semi-supervised

semisup_vat = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
)

semisup_vat_2way = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False),
)

semisup_vat_2way_entmin = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False,
                                      entropy_loss_coef=1),
)

semisup_vat_l = TrainerConfig(
    semisup_vat,
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=True),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True),
)

semisup_vat_l_2way = TrainerConfig(
    semisup_vat_l,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True, block_grad_on_clean=False),
)

semisup_vat_entmin = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, entropy_loss_coef=1),
)

semisup_cons_phtps20 = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
)

semisup_cons_warp1 = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(smooth_warp_attack, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
)

semisup_cons_phw2 = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phw_attack_1, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
)

semisup_cons_noise = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
)

semisup_cons_phtps20_r1w = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phtps_attack_20, step_count=0,
                                        loss=lambda p, c: losses.kl_div_ll(p.detach(), c),
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False),
)
semisup_cons_phtps20_1wa = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phtps_attack_20, step_count=0,
                                        loss=lambda p, c: losses.kl_div_ll(p, c.detach()),
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False),
)

semisup_cons_phtps20_entmin = TrainerConfig(
    semisup_cons_phtps20,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, entropy_loss_coef=1),
)

semisup_cons_phtps20_l = TrainerConfig(
    semisup_cons_phtps20,
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=True),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True),
)

semisup_cons_phtps20_entmin_l = TrainerConfig(
    semisup_cons_phtps20_l,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True, entropy_loss_coef=1),
)

semisup_cons_phtps20_seg = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False),
    data_loader_f=partial(
        vdu.simple_or_multi_data_loader,
        data_loader_f=vd.DataLoader,
        multi_data_loader_f=partial(vdu.morsic_semisup_data_loader,
                                    labeled_multiplier=lambda l, u: int(max(1, u / l))),
        num_workers=2,
    ),
)

# Semi-supervised: mean teacher

mean_teacher_custom_tps = TrainerConfig(
    partial(
        te.SemisupVAT, attack_f=partial(tps_warp_attack, step_count=0, loss=losses.kl_div_ll,
                                        output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False),
)

mean_teacher_custom_tps_weaker = TrainerConfig(
    partial(
        te.SemisupVAT,
        attack_f=partial(tps_warp_attack, initializer=pert.NormalInit({'offsets':
                                                                       (0, 0.05)}), projection=None,
                         step_count=0, loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False),
)

mean_teacher_custom_tps_more_weaker = TrainerConfig(
    partial(
        te.SemisupVAT,
        attack_f=partial(tps_warp_attack, initializer=pert.NormalInit({'offsets': (0, 0.02)}),
                         projection=pert.ScalingProjector({'offsets': 10}), step_count=0,
                         loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False),
)

mean_teacher_custom_tps_more_weaker_clean_teacher = TrainerConfig(
    partial(
        te.SemisupVAT,
        attack_f=partial(tps_warp_attack, initializer=pert.NormalInit({'offsets': (0, 0.02)}),
                         projection=pert.ScalingProjector({'offsets': 10}), step_count=0,
                         loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False, clean_teacher_input=True),
)

# Classification, CIFAR

resnet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=1e-4),
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128,
    jitter=jitter.CifarPadRandCropHFlip(),
)

resnet_cifar_cosine = TrainerConfig(
    resnet_cifar,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-4),
)

resnet_cifar_single_classifier_cv = TrainerConfig(
    # as in Computer Vision with a Single (Robust) Classiﬁer
    resnet_cifar,
    # overriding:
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=1e-4),
    epoch_count=350,
    batch_size=256,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[3 / 7, 5 / 7], gamma=0.1),
)

wrn_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    resnet_cifar,
    # overriding
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=5e-4),
)

irevnet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    resnet_cifar,
    # overriding
    # TODO: check lr. schedule and init
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=5e-4),
)

densenet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1608.06993
    resnet_cifar,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=True),
    epoch_count=100,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1),
    batch_size=64,
)

small_image_classifier = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=0),
    epoch_count=50,
    batch_size=64,
    jitter=jitter.CifarPadRandCropHFlip(),
)

ladder_densenet = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD, [dict(params='backbone.backbone', lr=1e-4)], lr=5e-4,
                               momentum=0.9, weight_decay=1e-4),
    lr_scheduler_f=partial(ScalableLR, lr_lambda=lambda p: (1 - p)**1.5),
    epoch_count=40,
    batch_size=4,
    jitter=jitter.SegRandHFlip(),
)

# Semantic segmentation, supervised

# https://github.com/orsic/swiftnet/blob/master/configs/rn18_single_scale.py
swiftnet_cityscapes = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.backbone', lr=1e-4, weight_decay=2.5e-5)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=14,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0),
)

swiftnet_irevnet_hybrid_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    eval_step=ts.DiscriminativeFlowSupervisedEvalStep(flow_end="backbone.backbone.concat",
                                                      gen_weight=1.),
    train_step=ts.DiscriminativeFlowSupervisedTrainStep(flow_end="backbone.backbone.concat",
                                                        gen_weight=1.),
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=5e-4),
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(256, 256), max_scale=2, overflow=0),
)

swiftnet_mo_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(optim.Adam, [
        dict(params=p, lr=1e-4, weight_decay=2.5e-5)
        for p in ["wrapped.backbone", "wrapped.logits"]], lr=4e-4, betas=(0.9, 0.99),
                               weight_decay=1e-4),
)

swiftnet_camvid = TrainerConfig(
    swiftnet_cityscapes,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=600,  # 600
    batch_size=12,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=2, overflow='half'),
)

semseg_basic = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam, [dict(params='backbone', lr=4e-4 / 5)], lr=4e-4,
                               betas=(0.9, 0.99), weight_decay=1e-4),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=40,
    batch_size=8,
    eval_batch_size=8,  # max 12?
    jitter=jitter.SegRandCropHFlip((768, 768)),
)

# Hybrid (discriminative, generative)

irevnet_hybrid_cifar = TrainerConfig(
    irevnet_cifar,
    # overriding
    # TODO: check lr. schedule and init
    eval_step=ts.DiscriminativeFlowSupervisedEvalStep(flow_end="backbone.concat", gen_weight=1.),
    train_step=ts.DiscriminativeFlowSupervisedTrainStep(flow_end="backbone.concat", gen_weight=1.),
    optimizer_f=partial(optim.Adamax, lr=1e-3, weight_decay=5e-4),
)

irevnet_adv_hybrid_cifar = TrainerConfig(
    irevnet_cifar,
    eval_step=ts.AdversarialDiscriminativeFlowSupervisedEvalStep(flow_end=('head', slice(2, None))),
    train_step=ts.AdversarialDiscriminativeFlowSupervisedTrainStep(flow_end=('head',
                                                                             slice(2, None))),
)

irevnet_at_hybrid_cifar = TrainerConfig(
    irevnet_cifar,
    eval_step=ts.AdversarialDiscriminativeFlowSupervisedEvalStep2(flow_end=('head',
                                                                            slice(2, None))),
    train_step=ts.AdversarialDiscriminativeFlowSupervisedTrainStep2(flow_end=('head',
                                                                              slice(2, None))),
)

# other

wrn_cifar_tent = TrainerConfig(
    wrn_cifar,
    # overriding
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=1e-4))

resnet_cifar_adversarial_esos = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    adversarial,
    attack_f=partial(madry_cifar10_attack, step_count=7),
    eval_attack_f=partial(madry_cifar10_attack, step_count=10, stop_on_success=True),
)

resnet_cifar_adversarial_multistep_esos = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar_adversarial_esos,
    # overriding
    train_step=ts.AdversarialTrainMultiStep(train_mode=False, update_period=8),
    epoch_count=25)

mnistnet = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification, optimizer_f=partial(optim.Adam, lr=1e-1, momentum=0.9, nesterov=True,
                                        weight_decay=1e-4), epoch_count=50,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8],
                           gamma=0.2), batch_size=128)

mnistnet_tent = TrainerConfig(
    classification,
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=1e-4),
    epoch_count=40,
    lr_scheduler_f=None,
    batch_size=100,
)
