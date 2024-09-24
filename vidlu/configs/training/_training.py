from torch import optim

from vidlu.transforms import jitter
from vidlu.optim.lr_schedulers import ScalableMultiStepLR, ScalableLR, CosineLR
from vidlu.optim import lr_shapes, OptimizerMaker
import vidlu.training.steps as ts
import vidlu.training.extensions as te

import vidlu.data as vd
import vidlu.data.utils as vdu
from vidlu.configs.robustness import *
from vidlu.utils.func import partial
from vidlu.modules import losses
from vidlu.modules.utils import proj

from .trainer_config import TrainerConfig

# Basic (train_step, eval_step)

supervised = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.supervised_train_step,
)

classification = TrainerConfig(
    supervised,
    loss=losses.nll_loss_l
)

# Adversarial training, basic

adversarial = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialStep(),
)

adversarial_combined = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialCombinedLossStep(),
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
    eval_step=ts.SemisupConsEvalStep(),
    train_step=ts.SemisupConsStep(),
)

semisup_multiscale_teacher = TrainerConfig(
    te.SemisupVAT,
    attack_f=attacks.DummyAttack,
    eval_step=ts.SemisupMultiScaleTeacherStep(eval=True),
    train_step=ts.SemisupMultiScaleTeacherStep(),
)

semisup_vat_2way = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupConsStep(block_grad_on_clean=False),
)

semisup_vat_2way_entmin = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupConsStep(block_grad_on_clean=False,
                                  entropy_loss_coef=1),
)

semisup_vat_l = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupConsStep(uns_loss_on_all=True),
    eval_step=ts.SemisupConsEvalStep(uns_loss_on_all=True),
)

semisup_vat_l_2way = TrainerConfig(
    semisup_vat_l,
    train_step=ts.SemisupConsStep(uns_loss_on_all=True, block_grad_on_clean=False),
)

semisup_vat_entmin = TrainerConfig(
    semisup_vat,
    train_step=ts.SemisupConsStep(entropy_loss_coef=1),
)

semisup_cons_phtps20 = TrainerConfig(  # doesn't work well unless ??? (not train_step.alpha=0.5)
    te.SemisupVAT,
    attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

rob_phtps20_halfbatch = TrainerConfig(  # doesn't work well unless ??? (not train_step.alpha=0.5)
    te.AdversarialTraining,
    attack_f=partial(phtps_attack_20, step_count=0, output_to_target=lambda x: x),
    train_step=ts.AdversarialStep(clean_proportion=0.5),
    eval_step=ts.supervised_eval_step,
)

rob_phtps20_fullbatch = TrainerConfig(  # doesn't work well unless ??? (not train_step.alpha=0.5)
    te.AdversarialTraining,
    attack_f=partial(phtps_attack_20, step_count=0, output_to_target=lambda x: x),
    train_step=ts.AdversarialStep(clean_proportion=0),
    eval_step=ts.supervised_eval_step,
)

semisup_cons_phtps20u = TrainerConfig(
    semisup_cons_phtps20,
    attack_f=partial(phtps_attack_20_unif, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
)

semisup_cons_warp1 = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(smooth_warp_attack, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_phw2 = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(phw_attack_1, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_noise = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_phtps20_r1w = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(phtps_attack_20, step_count=0,
                     loss=lambda p, c: losses.kl_div_ll(p.detach(), c),
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(block_grad_on_clean=False),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_phtps20_1wa = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(phtps_attack_20, step_count=0,
                     loss=lambda p, c: losses.kl_div_ll(p, c.detach()),
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(block_grad_on_clean=False),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_phtps20_entmin = TrainerConfig(
    semisup_cons_phtps20,
    train_step=ts.SemisupConsStep(entropy_loss_coef=1),
)

semisup_cons_phtps20_l = TrainerConfig(
    semisup_cons_phtps20,
    train_step=ts.SemisupConsStep(uns_loss_on_all=True),
    eval_step=ts.SemisupConsEvalStep(uns_loss_on_all=True),
)

semisup_cons_phtps20_entmin_l = TrainerConfig(
    semisup_cons_phtps20_l,
    train_step=ts.SemisupConsStep(uns_loss_on_all=True, entropy_loss_coef=1),
)

semisup_cons_ph3_seg = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(ph3_attack, output_to_target=lambda x: x, loss=losses.kl_div_ll),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_contr_ph3_seg = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(ph3_attack, output_to_target=lambda x: x, loss=losses.kl_div_ll),
    train_step=ts.SemisupContrastiveStepBase(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_tps20_seg = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(tps_attack_20, loss=losses.kl_div_ll, output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_tps20 = semisup_cons_tps20_seg

semisup_cons_ph20 = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(ph20_attack, loss=losses.kl_div_ll, output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_cutmix = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(cutmix_attack_21, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=ts.SemisupConsStep(),
    eval_step=ts.SemisupConsEvalStep(),
)

semisup_cons_phtps20_seg_morsic = TrainerConfig(
    semisup_cons_phtps20,
    data_loader_f=partial(
        vdu.auto_data_loader,
        dl_f=vd.DataLoader,
        multi_dl_f=partial(vdu.morsic_semisup_data_loader,
                           labeled_multiplier=lambda l, u: int(max(1, u / l))),
        num_workers=2,
    ),
)

# Semi-supervised: mean teacher

mean_teacher_custom_tps = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(tps_warp_attack, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    eval_step=ts.SemisupConsEvalStep(),
    train_step=ts.MeanTeacherStep(),
)

mean_teacher_custom_tps_weaker = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(tps_warp_attack, initializer=init.NormalInit({'offsets': (0, 0.05)}),
                     projection=None, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    eval_step=ts.SemisupConsEvalStep(),
    train_step=ts.MeanTeacherStep(),
)

mean_teacher_custom_tps_more_weaker = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(tps_warp_attack, initializer=init.NormalInit({'offsets': (0, 0.02)}),
                     projection=proj.ScalingProjector({'offsets': 10}), step_count=0,
                     loss=losses.kl_div_ll, output_to_target=lambda x: x),
    eval_step=ts.SemisupConsEvalStep(),
    train_step=ts.MeanTeacherStep(),
)

mean_teacher_custom_tps_more_weaker_clean_teacher = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(tps_warp_attack, initializer=init.NormalInit({'offsets': (0, 0.02)}),
                     projection=proj.ScalingProjector({'offsets': 10}), step_count=0,
                     loss=losses.kl_div_ll, output_to_target=lambda x: x),
    eval_step=ts.SemisupConsEvalStep(),
    train_step=ts.MeanTeacherStep(),
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
    lr_scheduler_f=partial(ScalableLR, lr_lambda=lambda p: (1 - p) ** 1.5),
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
    eval_batch_size=4,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

sn_cs_rand_scale_crop_ph3_jitter = jitter.Composition(
    jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                    scale_dist="log-uniform"),
    jitter.Photometric3())
sn_cs_rand_scale_crop_phtps_jitter = jitter.Composition(
    jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                    scale_dist="log-uniform"),
    jitter.PhTPS20())

sn_cs_rand_scale_crop_colorjitter1 = jitter.Composition(
    jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                    scale_dist="log-uniform"),
    jitter.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=1 / 6))

sn_cs_rand_scale_crop_colorjitter2 = jitter.Composition(
    jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                    scale_dist="log-uniform"),
    jitter.ColorJitter(brightness=(0.25, 2), contrast=0.25, saturation=0.25, hue=1 / 6))

sn_cs_rand_scale_crop_phtps_jitter = jitter.Composition(
    jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, scale_dist="log-uniform"),
    jitter.PhTPS20())

swiftnet_cityscapes1 = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(
        optim.Adam,
        [dict(params='backbone.backbone', lr=1e-4)],
        lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-6),
)

# 4 times smaller L2 regularization coefficient for semi-supervised learning
swiftnet_cityscapes_semi = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.backbone', lr=1e-4, weight_decay=2.5e-5 / 4)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=2.5e-5),
)

swiftnet_convnext_cityscapes_mg = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(
        optim.AdamW,
        [dict(params='backbone.backbone', lr=1e-4, weight_decay=0)],
        lr=4e-4, betas=(0.9, 0.99), weight_decay=0.),
    batch_size=10
)

swiftnet_pascal1 = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.backbone', lr=5e-5, weight_decay=2.5e-5)],
                               lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4),
    batch_size=12,
    eval_batch_size=4,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(320, 320), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

deeplabv2_pascal_ig = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD,
                               [dict(params='backbone.aspp', lr=2.5e-4)],
                               lr=2.5e-4, weight_decay=5e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=lambda e: 1 - e),
    epoch_count=80,  # TODO: 20000 iterations
    batch_size=5,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(320, 320), min_scale=0.5, max_scale=1.5,
                                           overflow=0, scale_dist="log-uniform"),
)

deeplabv2_pascal_french = TrainerConfig(
    classification,
    optimizer_f=partial(optim.SGD, lr=3e-5, weight_decay=5e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=274,  # TODO: 20000 iterations
    batch_size=10,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(321, 321), min_scale=0.5, max_scale=1.5,
                                           overflow=0, scale_dist="uniform"),
)

deeplabv3_pascal2 = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD,
                               [dict(params=['backbone.decoder', 'backbone.aspp'], lr=1e-2)],
                               lr=1e-3, weight_decay=1e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=274,  # TODO: 20000 iterations
    batch_size=5,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(320, 320), min_scale=0.5, max_scale=1.5,
                                           overflow=0, scale_dist="log-uniform"),
)

deeplabv3_pascal_cac = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD,
                               [dict(params=['backbone.decoder', 'backbone.aspp'], lr=1e-2)],
                               lr=1e-3, weight_decay=1e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=80,
    batch_size=8,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(320, 320), min_scale=0.5, max_scale=2,
                                           overflow=0, scale_dist="log-uniform"),
)

deeplabv3p_cityscapes_cac = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD,
                               [dict(params=['backbone.decoder', 'backbone.aspp'], lr=1e-1)],
                               lr=1e-2, weight_decay=1e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=250,
    batch_size=8,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(720, 720), min_scale=0.5, max_scale=1.5,
                                           overflow=0, scale_dist="log-uniform"),
)

deeplabv3p_cityscapes_ig = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD,
                               [dict(params=['backbone.decoder', 'backbone.aspp'], lr=1e-1)],
                               lr=1e-2, weight_decay=1e-4, momentum=0.9),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=250,
    batch_size=8,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), min_scale=0.5, max_scale=1.5,
                                           overflow=0, scale_dist="log-uniform"),
)

fc_hardnet = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=partial(
        optim.Adam,
        lr=2e-4, betas=(0.9, 0.99), weight_decay=5e-4),
)

swiftnet_cityscapes_semi2 = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(
        optim.Adam,
        [dict(params='backbone.backbone', lr=1e-4, weight_decay=1e-6)],
        lr=4e-4, betas=(0.9, 0.99), weight_decay=4e-6),
)

swiftnet_lamb_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(vo.LambAdam,
                               [dict(params='backbone.backbone', lr=1e-4, weight_decay=2.5e-5)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4),
)

swiftnet_mo_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(optim.Adam, [
        dict(params=lambda m: m.wrapped.fine_tune_params(), lr=4e-4 / 4, weight_decay=2.5e-5 / 4)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=2.5e-5),
)

ddrnet_cityscapes = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params=['backbone.spp', 'backbone.final_layer'], lr=4e-4,
                                     weight_decay=1e-4)],
                               lr=1e-4, betas=(0.9, 0.99), weight_decay=2.5e-5),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=14,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

deeplabv2_cityscapes_ig = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.aspp', lr=4e-4, weight_decay=1e-4)], lr=1e-4,
                               betas=(0.9, 0.99), weight_decay=2.5e-5),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=250,
    batch_size=10,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

swiftnet_cityscapes_halfres = TrainerConfig(
    swiftnet_cityscapes,
    batch_size=8,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=1.5, overflow=0,
                                           scale_dist="log-uniform"))

deeplabv2_cityscapes_halfres_mva = TrainerConfig(
    swiftnet_cityscapes_halfres,
    batch_size=4,
    optimizer_f=deeplabv2_cityscapes_ig.optimizer_f,
)

deeplabv2_cityscapes_halfres_mva2 = TrainerConfig(
    classification,
    swiftnet_cityscapes_halfres,
    optimizer_f=partial(optim.Adam, lr=3e-5),
    lr_scheduler_f=partial(ScalableLR, func=partial(lr_shapes.poly, power=0.9)),
    epoch_count=100,
    batch_size=4,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

swiftnet_irevnet_hybrid_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    eval_step=ts.DiscriminativeFlowSupervisedEvalStep(flow_end="backbone.backbone.concat",
                                                      gen_weight=1.),
    train_step=ts.DiscriminativeFlowSupervisedTrainStep(flow_end="backbone.backbone.concat",
                                                        gen_weight=1.),
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=5e-4),
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(256, 256), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
)

swiftnet_camvid = TrainerConfig(
    swiftnet_cityscapes,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=600,  # 600
    batch_size=12,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=2, overflow='half',
                                           scale_dist="log-uniform"),
)

swiftnet_camvid_semi = TrainerConfig(
    swiftnet_camvid,
    optimizer_f=swiftnet_cityscapes_semi.optimizer_f,
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

maskswifter_cityscapes = TrainerConfig(
    classification,
    train_step=ts.MaskFormerSegStep(),
    eval_step=None,
    optimizer_f=OptimizerMaker(optim.Adam, [dict(params='backbone', lr=1e-4, weight_decay=2.5e-5)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=8,
    eval_batch_size=4,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist="log-uniform"),
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
