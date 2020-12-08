from torch import optim

from vidlu.data import Record
from vidlu.modules import get_submodule
from vidlu.transforms import jitter
from vidlu.utils.collections import NameDict
from vidlu.utils.func import params, Required
from vidlu.optim.lr_schedulers import ScalableMultiStepLR, ScalableLambdaLR, CosineLR
import vidlu.training.steps as ts
import vidlu.training.extensions as te

from vidlu.configs.robustness import *


# Optimizer maker


class OptimizerMaker:
    """A type for storing all optimizer information without depending on a
    model instance by storing module names instead of parameters.

    Calling an object of this type with a model creates an optimizer instance.

    Args:
        optimizer_f: PyTorch optimizer factory (or constructor).
        params (List[Mapping]): A list of dictionaries in the same format as
            for PyTorch optimizers except for module names instead of
            parameters.
        ignore_remaining_params: Specifies whether unlisted parameters should be
            ignored instead of being optimized.
        **kwargs: Keyword arguments for the optimizer.
    """

    def __init__(self, optimizer_f, params, ignore_remaining_params=False, **kwargs):
        self.optimizer_f, self.params, self.kwargs = optimizer_f, params, kwargs
        self.ignore_remaining_params = ignore_remaining_params

    def __call__(self, model):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"The model argument should be a nn.Module, not {type(model)}.")
        params = [{**d, 'params': tuple(get_submodule(model, d['params']).parameters())}
                  for d in self.params]
        params_lump = set(p for d in params for p in d['params'])
        remaining_params = () if self.ignore_remaining_params \
            else tuple(p for p in model.parameters() if p not in params_lump)
        return self.optimizer_f([{'params': remaining_params}] + params, **self.kwargs)


# Trainer config

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

    def normalized(self):
        """Creates an equivalent TrainerConfig where arguments for extensions
         are bound to corresponding extension factories and removed from the
         main namespace.

        A normalized TrainerConfig can be given to the Trainer constructor.

        Example:
            >>> from vidlu.training.trainers import Trainer
            >>> tc: TrainerConfig(...)
            >>> trainer = Trainer(**tc.normalized())
        """
        result = TrainerConfig(**self)
        arg_name_to_ext = dict()
        ext = []
        for ext_f in result.extension_fs:
            names = tuple(params(ext_f).keys())
            values = [result.pop(name, Required) for name in names]
            args = {k: v for k, v in zip(names, values) if v is not Required}
            ext.append(partial(ext_f, **args) if len(args) > 0 else ext_f)
            for name in names:
                if name in arg_name_to_ext:
                    raise RuntimeError(f'Multiple extension factories have a parameter "{name}".')
                arg_name_to_ext[name] = ext_f
        result.extension_fs = ext
        return result


# Extend output

def classification_extend_output(output):
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor):
        raise ValueError("The output or its first element must be a `torch.Tensor`"
                         + " representing class scores.")
    return logits, Record(output=logits, full_output=output, log_probs_=lambda: logits.log_softmax(1),
                          probs_=lambda r: r.log_probs.exp(),
                          hard_prediction_=lambda: logits.argmax(1))


# Trainer configs ##################################################################################

supervised = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.supervised_train_step)

adversarial = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialTrainStep())

adversarial_combined = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialCombinedLossTrainStep())

adversarial_free = TrainerConfig(
    te.AdversarialTraining,
    eval_step=ts.AdversarialEvalStep(),
    train_step=ts.AdversarialTrainMultiStep())

vat = TrainerConfig(
    adversarial,
    train_step=ts.VATTrainStep(),
    eval_step=ts.AdversarialEvalStep(virtual=True),
    attack_f=attacks.VATAttack)

semisup_vat = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False))

semisup_vat_2way = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False))

semisup_vat_2way_entmin = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False,
                                      block_grad_on_clean=False,
                                      entropy_loss_coef=1))

semisup_vat_l = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=True),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True))

semisup_vat_l_2way = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=True),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True, block_grad_on_clean=False))

semisup_vat_entmin = TrainerConfig(
    partial(te.SemisupVAT, attack_f=attacks.VATAttack),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, entropy_loss_coef=1))

semisup_cons_phtps20 = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                             output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False))

semisup_cons_phtps20 = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                             output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False))

semisup_cons_noise = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                             output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False))

semisup_cons_phtps20_r1w = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(phtps_attack_20, step_count=0, loss=lambda p, c: losses.kl_div_ll(p.detach(), c),
                             output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False))

semisup_cons_phtps20_1wa = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(phtps_attack_20, step_count=0, loss=lambda p, c: losses.kl_div_ll(p, c.detach()),
                             output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, block_grad_on_clean=False))

semisup_cons_phtps20_entmin = TrainerConfig(
    semisup_cons_phtps20,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False, entropy_loss_coef=1))

semisup_cons_phtps20_l = TrainerConfig(
    semisup_cons_phtps20,
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=True),
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True))

semisup_cons_phtps20_entmin_l = TrainerConfig(
    semisup_cons_phtps20_l,
    train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=True, entropy_loss_coef=1))

mean_teacher_custom_tps = TrainerConfig(
    partial(te.SemisupVAT, attack_f=partial(tps_warp_attack, step_count=0,
                                            loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False))

mean_teacher_custom_tps_weaker = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(tps_warp_attack,
                             initializer=pert.NormalInit({'offsets': (0, 0.05)}),
                             projection=None,
                             step_count=0, loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False))

mean_teacher_custom_tps_more_weaker = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(tps_warp_attack,
                             initializer=pert.NormalInit({'offsets': (0, 0.02)}),
                             projection=pert.ScalingProjector({'offsets': 10}),
                             step_count=0, loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False))

mean_teacher_custom_tps_more_weaker_clean_teacher = TrainerConfig(
    partial(te.SemisupVAT,
            attack_f=partial(tps_warp_attack,
                             initializer=pert.NormalInit({'offsets': (0, 0.02)}),
                             projection=pert.ScalingProjector({'offsets': 10}),
                             step_count=0, loss=losses.kl_div_ll, output_to_target=lambda x: x)),
    eval_step=ts.SemisupVATEvalStep(consistency_loss_on_labeled=False),
    train_step=ts.MeanTeacherTrainStep(consistency_loss_on_labeled=False,
                                       clean_teacher_input=True))

classification = TrainerConfig(
    supervised,
    extend_output=classification_extend_output)

autoencoder = TrainerConfig(
    eval_step=ts.supervised_eval_step,
    train_step=ts.autoencoder_train_step)

resnet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=1e-4),
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128,
    jitter=jitter.CifarPadRandCropHFlip())

resnet_cifar_cosine = TrainerConfig(
    resnet_cifar,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-4))

resnet_cifar_single_classifier_cv = TrainerConfig(
    # as in Computer Vision with a Single (Robust) Classiﬁer
    resnet_cifar,
    # overriding:
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=1e-4),
    epoch_count=350,
    batch_size=256,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[3 / 7, 5 / 7], gamma=0.1))

wrn_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    resnet_cifar,
    # overriding
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=5e-4))

irevnet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    resnet_cifar,
    # overriding
    # TODO: check lr. schedule and init
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=5e-4))

irevnet_hybrid_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1605.07146
    irevnet_cifar,
    # overriding
    # TODO: check lr. schedule and init
    eval_step=ts.DiscriminativeFlowSupervisedEvalStep(flow_end="backbone.concat", gen_weight=1.),
    train_step=ts.DiscriminativeFlowSupervisedTrainStep(flow_end="backbone.concat", gen_weight=1.),
    optimizer_f=partial(optim.Adamax, lr=1e-3, weight_decay=5e-4))

densenet_cifar = TrainerConfig(  # as in www.arxiv.org/abs/1608.06993
    resnet_cifar,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=True),
    epoch_count=100,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1),
    batch_size=64)

small_image_classifier = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=0),
    epoch_count=50,
    batch_size=64,
    jitter=jitter.CifarPadRandCropHFlip())

ladder_densenet = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.SGD, [dict(params='backbone.backbone', lr=1e-4)],
                               lr=5e-4, momentum=0.9, weight_decay=1e-4),
    lr_scheduler_f=partial(ScalableLambdaLR, lr_lambda=lambda p: (1 - p) ** 1.5),
    epoch_count=40,
    batch_size=4,
    jitter=jitter.SegRandHFlip())

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
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0))

swiftnet_irevnet_hybrid_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    eval_step=ts.DiscriminativeFlowSupervisedEvalStep(flow_end="backbone.backbone.concat", gen_weight=1.),
    train_step=ts.DiscriminativeFlowSupervisedTrainStep(flow_end="backbone.backbone.concat", gen_weight=1.),
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=5e-4),
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(256, 256), max_scale=2, overflow=0))

swiftnet_mo_cityscapes = TrainerConfig(
    swiftnet_cityscapes,
    optimizer_f=OptimizerMaker(
        optim.Adam,
        [dict(params=p, lr=1e-4, weight_decay=2.5e-5)
         for p in ["wrapped.backbone", "wrapped.logits"]],
        lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4))

swiftnet_camvid = TrainerConfig(
    swiftnet_cityscapes,
    # overriding:
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=600,  # 600
    batch_size=12,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=2, overflow='half'))

semseg_basic = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam, [dict(params='backbone', lr=4e-4 / 5)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=40,
    batch_size=8,
    eval_batch_size=8,  # max 12?
    jitter=jitter.SegRandCropHFlip((768, 768)))

# other

mnistnet = TrainerConfig(  # as in www.arxiv.org/abs/1603.05027
    classification,
    optimizer_f=partial(optim.Adam, lr=1e-1, momentum=0.9, nesterov=True, weight_decay=1e-4),
    epoch_count=50,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128)

mnistnet_tent = TrainerConfig(
    classification,
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=1e-4),
    epoch_count=40,
    lr_scheduler_f=None,
    batch_size=100)

# special

wrn_cifar_tent = TrainerConfig(
    wrn_cifar,
    # overriding
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=1e-4))

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
