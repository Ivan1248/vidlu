from functools import partial
import contextlib
from dataclasses import dataclass

import torch
from torch import optim

from vidlu.data import Record
from vidlu.modules import get_submodule
from vidlu.transforms import jitter
from vidlu.utils.func import default_args, params, Empty
from vidlu.utils.misc import fuse, dict_difference
from vidlu.utils.collections import NameDict
from vidlu.utils.torch import fuse_tree_batches, disable_tracking_bn_stats
from .adversarial import attacks
from .lr_schedulers import ScalableMultiStepLR, ScalableLambdaLR, CosineLR
import torch.nn.functional as F


# Get outputses

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


# Training/evaluation steps ########################################################################


## Supervised

@torch.no_grad()
def supervised_eval_step(trainer, batch):
    trainer.model.eval()
    x, y = trainer.prepare_batch(batch)
    output, other_outputs = trainer.extend_output(trainer.model(x))
    loss = trainer.loss(output, y)
    return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item())


def _supervised_train_step_x_y(trainer, x, y):
    trainer.model.train()
    trainer.optimizer.zero_grad()
    output, other_outputs = trainer.extend_output(trainer.model(x))
    loss = trainer.loss(output, y)
    loss.backward()
    trainer.optimizer.step()
    return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item())


def supervised_train_step(trainer, batch):
    return _supervised_train_step_x_y(trainer, *trainer.prepare_batch(batch))


## Supervised multistep

class SupervisedTrainMultiStep:
    """A training step where each batch is used multiple times consecutively.

    Args:
        repeat_count (int): Number of times each batch is repeated.
    """

    def __init__(self, repeat_count):
        self.repeat_count = repeat_count

    def __call__(self, trainer, batch):
        trainer.model.train()
        for i in range(self.repeat_count):
            with disable_tracking_bn_stats(trainer.model) if i == 0 else contextlib.suppress():
                trainer.optimizer.zero_grad()
                x, y = trainer.prepare_batch(batch)
                output, other_outputs = trainer.extend_output(trainer.model(x))
                loss = trainer.loss(output, y)
                loss.backward()
                trainer.optimizer.step()
                if i == 0:
                    initial = dict(output=output, other_outputs=other_outputs, loss=loss.item())
        final = dict(output=output, other_outputs=other_outputs, loss=loss.item())
        return NameDict(x=x, target=y, **initial, **{f"{k}_post": v for k, v in final.items()})


class SupervisedSlidingBatchTrainStep:
    """A training step where the batch slides through training examples with a
    stride smaller than batch size, making each example used
    `batch_size//stride` times consecutively. but in batches differing in
    `2*stride` examples (`stride` examples removed from the beginning, `stride` examples
    added to the end).

    This should hopefully reduce overfitting with respect to
    `SupervisedTrainMultiStep`, where each batch is repeated `repeat_count`
    (equivalent to `steps_per_batch` here).

    Args:
        stride (int): Number of examples tha batch is shifted in each substep.
        steps_per_batch (int): Number of times the batch is shifted per per the
            length of a batch. If this is provided, `stride` is automatically
            computed as `stride=batch_size//steps_per_batch`.
        reversed (bool, True): Use same batches but in reversed order in each
            multi-step. This is True by default because it enables more variety
            between consecutive batches during training and doesn't make the
            returned predictions too optimistic.

    The arguments `stride` and `steps_per_batch` are mutually exclusive, i.e.
    only 1 has to be provided.
    """

    def __init__(self, steps_per_batch=None, stride=None, reversed=True):
        self.prev_x_y = None
        self.start_index = 0
        if (stride is None) == (steps_per_batch is None):
            raise ValueError("Either `stride` or `steps_per_batch` should be provided.")
        self.steps_per_batch = steps_per_batch
        self.stride = stride
        self.reversed = reversed

    def __call__(self, trainer, batch):
        trainer.model.train()
        n = len(batch[0])
        stride = self.stride or n // self.steps_per_batch
        x_y = trainer.prepare_batch(batch)

        if self.prev_x_y is None:  # the first batch
            self.prev_x_y = x_y
            self.start_index = stride
            return _supervised_train_step_x_y(trainer, *x_y)

        x, y = [torch.cat([a, b], dim=0) for a, b in zip(self.prev_x_y, x_y)]
        starts = list(range(self.start_index, len(x) - n + 1, stride))
        if self.reversed:
            starts = list(reversed(starts))
        result = None
        for i in starts:
            with disable_tracking_bn_stats(trainer.model) if i == starts[0] \
                    else contextlib.suppress():
                inter_x, inter_y = [a[i:i + n] for a in (x, y)]
                trainer.optimizer.zero_grad()
                output, other_outputs = trainer.extend_output(trainer.model(inter_x))
                loss = trainer.loss(output, inter_y)
                loss.backward()
                trainer.optimizer.step()
                result = result or NameDict(x=inter_x, target=inter_y, output=output,
                                            other_outputs=other_outputs, loss=loss.item())
        self.start_index = starts[0 if self.reversed else -1] + stride - len(self.prev_x_y[0])
        # Data for the last inter-batch iteration is returned
        # The last inter-batch is not necessarily `batch`
        # WARNING: Training performance might be too optimistic if reversed=False
        return result


## Supervised accumulated batch

class SupervisedTrainAcummulatedBatchStep:
    def __init__(self, batch_split_factor, reduction='mean'):
        super().__init__()
        self.batch_split_factor = batch_split_factor
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError("reduction is not in {'mean', 'sum'}.")

    def __call__(self, trainer, batch):
        if batch.shape(0) % self.batch_split_factor != 0:
            raise RuntimeError(f"Batch size ({batch.shape(0)}) is not a multiple"
                               + f" of batch_split_factor ({self.batch_split_factor}).")
        trainer.model.train()
        trainer.optimizer.zero_grad()
        x, y = trainer.prepare_batch(batch)
        xs, ys = [b.split(len(x) // self.batch_split_factor, dim=0) for b in [x, y]]
        outputs, other_outputses = [], []
        total_loss = None
        for x, y in zip(xs, ys):
            output, other = trainer.extend_output(trainer.model(x))
            outputs.append(output)
            other_outputses.append(other_outputses)
            loss = trainer.loss(output, y)
            if self.reduction == 'mean':
                loss /= len(x)
            loss.backward()
            if total_loss is None:
                total_loss = loss.detach().clone()
            else:
                total_loss += loss.detach()
        trainer.optimizer.step()
        return NameDict(x=x, target=y, output=torch.cat(outputs, dim=0),
                        other_outputs=fuse_tree_batches(other_outputses), loss=total_loss.item())


# Adversarial

class CleanResultCallback:
    def __init__(self, extend_output):
        self.extend_output = extend_output
        self.result = None

    def __call__(self, r):
        if self.result is None:
            self.result = NameDict({
                **vars(r),
                **dict(zip(("output", "other_outputs"), self.extend_output(r.output)))})


def first_output_callback(output_ref: list):
    def backward_callback(r):
        nonlocal output_ref
        if len(output_ref) == 0:
            output_ref.append(r)


class AdversarialTrainStep:
    """ Adversarial training step with the option to keep a the batch partially
    clean.

    It calls `trainer.model.eval()`, `trainer.attack.perturb` with
    `1 - clean_proportion` of input examples, and the corresponding labels
    if `virtual` is `False`. Then makes a training step with adversarial
    examples mixed with clean examples, depending on `clean_proportion`.

    'mean' reduction is assumed for the loss.


    Args:
        clean_proportion (float): The proportion of input examples that should
            not be turned into adversarial examples in each step. Default: 0.
        virtual (bool): A value determining whether virtual adversarial examples
            should be used (using the predicted label).
    """

    def __init__(self, clean_proportion=0, virtual=False):
        self.clean_proportion = clean_proportion
        self.virtual = virtual

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)
        cln_count = round(self.clean_proportion * len(x))
        cln_proportion = cln_count / len(x)
        split = lambda a: (a[:cln_count], a[cln_count:])
        (x_c, x_a), (y_c, y_a) = split(x), split(y)

        crc = CleanResultCallback(trainer.extend_output)
        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_adv = trainer.attack.perturb(x_a, *(() if self.virtual else (y[cln_count:],)),
                                       backward_callback=crc)
        trainer.model.train()
        trainer.optimizer.zero_grad()

        output, other_outputs = trainer.extend_output(trainer.model(torch.cat((x_c, x_adv), dim=0)))
        output_c, output_adv = split(output)
        loss_adv = trainer.loss(output_adv, y_a)
        loss_c = trainer.loss(output_c, y_c) if len(y_c) > 0 else 0
        (cln_proportion * loss_c + (1 - cln_proportion) * loss_adv).backward()

        trainer.optimizer.step()

        other_outputs_adv = NameDict({k: a[cln_count:] for k, a in other_outputs.items()})
        return NameDict(x=x, output=crc.result.output, target=y,
                        other_outputs=crc.result.other_outputs, loss=crc.result.loss,
                        x_adv=x_adv, target_adv=y_a, output_adv=output_adv,
                        other_outputs_adv=other_outputs_adv, loss_adv=loss_adv.item())


class AdversarialCombinedLossTrainStep:
    """A training step that first performs an optimization on an weighted
    combination of the standard loss and the adversarial loss.

    Args:
        use_attack_loss (bool): A value determining whether the loss function
            from the attack (`trainer.attack.loss`) should be used as the
            adversarial loss instead of the standard loss function
            (`trainer.loss`).
        clean_weight (float): The weight of the clean loss, a number
            normally between 0 and 1. Default: 0.5.
        adv_weight (float): The weight of the clean loss, a number normally
            between 0 and 1. If no value is provided (`None`), it is computed as
            `1 - clean_loss_weight`.
        virtual (bool): A value determining whether virtual adversarial examples
            should be used (using the predicted label).
    """

    def __init__(self, use_attack_loss, clean_weight=0.5, adv_weight=None, virtual=False):
        self.use_attack_loss = use_attack_loss
        self.clean_loss_weight = clean_weight
        self.adv_loss_weight = 1 - clean_weight if adv_weight is None else adv_weight
        self.virtual = virtual

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)

        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_adv = trainer.attack.perturb(x, *(() if self.virtual else (y,)))

        trainer.model.train()
        trainer.optimizer.zero_grad()

        output_c, other_outputs_c = trainer.extend_output(trainer.model(x))
        loss_c = trainer.loss(output_c, y)
        output_adv, other_outputs_adv = trainer.extend_output(trainer.model(x_adv))
        loss_adv = (trainer.attack if self.use_attack_loss else trainer).loss(output_adv, y)
        loss = self.clean_loss_weight * loss_c + self.adv_loss_weight * loss_adv
        loss.backward()

        trainer.optimizer.step()

        return NameDict(x=x, output=output_c, target=y, other_outputs=other_outputs_c, loss=loss_c,
                        x_adv=x_adv, output_adv=output_adv, other_outputs_adv=other_outputs_adv,
                        loss_adv=loss_adv.item())


class AdversarialBiTrainStep:
    """A training step that first performs an optimization step on a clean batch
    and then on the batch turned into adversarial examples.

    Args:
        virtual (bool): A value determining whether virtual adversarial examples
            should be used (using the predicted label).
    """

    def __init__(self, virtual=False):
        self.virtual = virtual

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)
        clean_result = _supervised_train_step_x_y(trainer, x, y)
        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_adv = trainer.attack.perturb(x, *(() if self.virtual else (y,)))
        result_adv = _supervised_train_step_x_y(trainer, x_adv, y)
        return NameDict(x=x, output=clean_result.output, target=y,
                        other_outputs=clean_result.other_outputs, loss=clean_result.loss,
                        x_adv=x_adv, output_adv=result_adv.output,
                        other_outputs_adv=result_adv.other_outputs, loss_adv=result_adv.loss)


class AdversarialTrainMultiStep:
    def __init__(self, update_period=1, virtual=False, train_mode=True):
        self.update_period = update_period
        self.virtual = virtual
        self.train_mode = train_mode

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)
        clean_result = None

        def step(r):
            nonlocal clean_result, x, self
            if clean_result is None:
                clean_result = r
            if r.step % self.update_period == 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

        (trainer.model.train if self.train_mode else trainer.model.eval)()
        with disable_tracking_bn_stats(trainer.model) if self.train_mode else contextlib.suppress():
            x_adv = trainer.attack.perturb(x, *(() if self.virtual else (y,)),
                                           backward_callback=step)

        trainer.model.train()
        trainer.optimizer.zero_grad()
        output_adv, other_outputs_adv = trainer.extend_output(trainer.model(x_adv))
        loss_adv = trainer.loss(output_adv, y)
        loss_adv.backward()
        trainer.optimizer.step()

        return NameDict(x=x, target=y, output=clean_result.output,
                        other_outputs=trainer.extend_output(clean_result.output)[1],
                        loss=clean_result.loss, x_adv=x_adv, output_adv=output_adv,
                        other_outputs_adv=other_outputs_adv, loss_adv=loss_adv.item())


class VATTrainStep:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)

        trainer.model.train()
        output, other_outputs = trainer.extend_output(trainer.model(x))
        loss = trainer.loss(output, y)
        loss.backward()
        trainer.optimizer.step()
        clean_grads = {k: p.grad.detach().clone() for k, p in trainer.model.named_parameters()}

        trainer.model.eval()  # NOTE THIS
        x_adv = trainer.attack.perturb(x, output)

        trainer.model.train()
        with disable_tracking_bn_stats(trainer.model):
            trainer.optimizer.zero_grad()
            output_adv, other_outputs_adv = trainer.extend_output(trainer.model(x_adv))
            loss_adv = self.alpha * trainer.attack.loss(output_adv, y)
            (self.alpha * loss_adv).backward()

        for k, p in trainer.model.named_parameters():
            p.grad += clean_grads[k]
        trainer.optimizer.step()

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_adv=x_adv, output_adv=output_adv, other_outputs_adv=other_outputs_adv,
                        loss_adv=loss_adv.item())


def adversarial_eval_step(trainer, batch):
    trainer.model.eval()

    x, y = trainer.prepare_batch(batch)

    with torch.no_grad():
        output, other_outputs = trainer.extend_output(trainer.model(x))
        loss = trainer.loss(output, y)

    x_adv = trainer.eval_attack.perturb(x, y)
    with torch.no_grad():
        output_adv, other_outputs_adv = trainer.extend_output(trainer.model(x_adv))
        loss_adv = trainer.loss(output_adv, y)

    return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                    x_adv=x_adv, output_adv=output_adv, other_outputs_adv=other_outputs_adv,
                    loss_adv=loss_adv.item())


"""
def adversarial_train_step(trainer, batch):
    trainer.model.train()
    x, y = trainer.prepare_batch(batch)

    trainer.optimizer.zero_grad()
    output, other_outputs = trainer.extend_output(trainer.model(x))
    loss = trainer.loss(output, y)
    loss.backward()
    trainer.optimizer.step()

    x_adv = trainer.attack.perturb(x, y)
    trainer.optimizer.zero_grad()
    output_adv, other_outputs_adv = trainer.extend_output(trainer.model(x_adv))
    loss_adv = trainer.loss(output_adv, y)
    loss_adv.backward()
    trainer.optimizer.step()

    return NameDict(x=x, y=y, output=output, target=y, other_outputs=other_outputs,
                    loss=loss.item(),
                    x_adv=x_adv, output_adv=output_adv, other_outputs_adv=other_outputs_adv,
                    loss_adv=loss_adv.item())
"""


# Autoencoder

def autoencoder_train_step(trainer, batch):
    trainer.model.train()
    trainer.optimizer.zero_grad()
    x = trainer.prepare_batch(batch)[0]
    x_r = trainer.model(x)
    loss = trainer.loss(x_r, x)
    loss.backward()
    trainer.optimizer.step()
    return NameDict(x_r=x_r, x=x, loss=loss.item())


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

# Hyperparameter sets ##############################################################################


supervised = dict(
    eval_step=supervised_eval_step,
    train_step=supervised_train_step)

adversarial = dict(
    eval_step=adversarial_eval_step,  # TODO
    train_step=AdversarialTrainStep())

adversarial_free = dict(
    eval_step=adversarial_eval_step,  # TODO
    train_step=AdversarialTrainMultiStep())

vat = dict(
    eval_step=adversarial_eval_step,
    train_step=AdversarialCombinedLossTrainStep(use_attack_loss=True, clean_weight=1, adv_weight=1),
    attack_f=attacks
)

classification = dict(
    **supervised,
    extend_output=classification_extend_output)

autoencoder = dict(
    eval_step=supervised_eval_step,
    train_step=autoencoder_train_step)

resnet_cifar = dict(  # as in www.arxiv.org/abs/1603.05027
    **classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=200,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128,
    jitter=jitter.CifarPadRandomCropHFlip())

resnet_cifar_cosine = fuse(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    overriding=dict(lr_scheduler_f=partial(CosineLR, eta_min=1e-4)))

resnet_cifar_single_classifier_cv = fuse(  # as in Computer Vision with a Single (Robust) Classiﬁer
    resnet_cifar,
    overriding=dict(
        optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=Empty),
        epoch_count=350,
        batch_size=256,
        lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[3 / 7, 5 / 7], gamma=0.1)))

wrn_cifar = fuse(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    overriding=dict(weight_decay=5e-4))

densenet_cifar = dict(  # as in www.arxiv.org/abs/1608.06993
    **classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, weight_decay=Empty, nesterov=True),
    epoch_count=100,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1),
    batch_size=64,
    jitter=jitter.CifarPadRandomCropHFlip())

small_image_classifier = dict(  # as in www.arxiv.org/abs/1603.05027
    **classification,
    weight_decay=0,
    optimizer_f=partial(optim.SGD, lr=1e-2, momentum=0.9),
    epoch_count=50,
    batch_size=64,
    jitter=jitter.CifarPadRandomCropHFlip())

ladder_densenet = dict(
    **classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.SGD, lr=5e-4, momentum=0.9, weight_decay=Empty),
    lr_scheduler_f=partial(ScalableLambdaLR, lr_lambda=lambda p: (1 - p) ** 1.5),
    epoch_count=40,
    batch_size=4,
    optimizer_maker=FineTuningOptimizerMaker({'backbone.backbone': 1 / 4}),
    jitter=jitter.SegRandomHFlip())

swiftnet = dict(
    **classification,
    weight_decay=1e-4,  # za imagenet 4 puta manje, tj. 16 puta uz množenje r korakom učenja
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=Empty),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=14,
    eval_batch_size=8,
    optimizer_maker=FineTuningOptimizerMaker({'backbone.backbone': 1 / 4}),
    jitter=jitter.SegRandomCropHFlip((768, 768)))

swiftnet_camvid = fuse(
    swiftnet,
    overriding=dict(
        lr_scheduler_f=partial(CosineLR, eta_min=1e-7),
        epoch_count=600,  # 600
        batch_size=12,
        jitter=jitter.SegRandomCropHFlip((448, 448))))  # 448

swiftnet_camvid_scratch = fuse(
    swiftnet_camvid,
    overriding=dict(
        epoch_count=900,
        optimizer_maker=None))

semseg_basic = dict(
    **classification,
    weight_decay=1e-4,
    optimizer_f=partial(optim.Adam, lr=4e-4, betas=(0.9, 0.99), weight_decay=Empty),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=40,
    batch_size=8,
    eval_batch_size=8,  # max 12?
    optimizer_maker=FineTuningOptimizerMaker({'backbone': 1 / 4}),
    jitter=jitter.SegRandomCropHFlip((768, 768)))

mnistnet = dict(  # as in www.arxiv.org/abs/1603.05027
    **classification,
    optimizer_f=partial(optim.SGD, lr=1e-1, momentum=0.9, nesterov=True, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=50,
    lr_scheduler_f=partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2),
    batch_size=128)

mnistnet_tent = dict(
    **classification,
    optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=Empty),
    weight_decay=1e-4,
    epoch_count=40,
    lr_scheduler_f=None,
    batch_size=100)

mnistnet_tent_eval_attack = partial(attacks.PGDAttack,
                                    eps=0.3,
                                    step_size=0.1,
                                    grad_preprocessing='sign',
                                    step_count=40,  # TODO: change
                                    clip_bounds=(0, 1),
                                    stop_on_success=True)

## special

wrn_cifar_tent = fuse(
    wrn_cifar,
    overriding=dict(optimizer_f=partial(optim.Adam, lr=1e-3, weight_decay=Empty),
                    weight_decay=1e-4))

resnet_cifar_adversarial_esos = fuse(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar,
    dict(attack_f=partial(madry_cifar10_attack, step_count=7),
         eval_attack_f=partial(madry_cifar10_attack, step_count=10, stop_on_success=True)),
    overriding=adversarial)

resnet_cifar_adversarial_multistep_esos = fuse(  # as in www.arxiv.org/abs/1603.05027
    resnet_cifar_adversarial_esos,
    overriding=dict(train_step=AdversarialTrainMultiStep(train_mode=False, update_period=8),
                    epoch_count=25))
