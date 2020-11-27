from contextlib import suppress as ctx_suppress
import dataclasses as dc
from functools import partial
import typing as T
from torch import nn
import copy

import numpy as np
import torch
from torch import nn

from vidlu.data import BatchTuple
from vidlu.utils.collections import NameDict
from vidlu.torch_utils import (concatenate_tensors_trees, switch_training,
                               batchnorm_stats_tracking_off)
import vidlu.modules.losses as vml
import vidlu.modules.elements as vme
import vidlu.modules.utils as vmu

from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj


# Training/evaluation steps ########################################################################

@dc.dataclass
class TrainerStep:  # TODO: use to reduce
    outputs: dict = dc.field(
        default_factory=lambda: dict(zip(*[["x", "target", "output", "other_outputs", "loss"]] * 2)))

    def __call__(self, trainer, batch):
        output = self.run(trainer, batch)
        return NameDict({self.outputs[k]: v for k, v in output.items() if k in self.outputs})


# Supervised

def do_optimization_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@torch.no_grad()
def supervised_eval_step(trainer, batch):
    trainer.model.eval()
    x, y = trainer.prepare_batch(batch)
    output, other_outputs = trainer.extend_output(trainer.model(x))
    loss = trainer.loss(output, y).mean()
    return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item())


def _supervised_train_step_x_y(trainer, x, y):
    trainer.model.train()
    output, other_outputs = trainer.extend_output(trainer.model(x))
    loss = trainer.loss(output, y).mean()
    do_optimization_step(trainer.optimizer, loss)
    return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item())


def supervised_train_step(trainer, batch):
    return _supervised_train_step_x_y(trainer, *trainer.prepare_batch(batch))


@dc.dataclass
class ClassifierEnsembleEvalStep:
    model_iter: T.Callable[[nn.Module], T.Iterator[nn.Module]]
    combine: T.Callable[[T.Sequence], torch.Tensor] = lambda x: torch.stack(x).mean(0)

    def __call__(self, trainer, batch):
        trainer.model.eval()
        x, y = trainer.prepare_batch(batch)
        output = self.combine(model(x) for model in self.model_iter(trainer.model))
        _, other_outputs = trainer.extend_output(output)
        loss = trainer.loss(output, y).mean()
        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item())


# generative flows and discriminative hybrids

def dequantize(x, bin_count, scale=1):
    return x + scale / bin_count * torch.rand_like(x)


def call_flow(model, x, end=None):
    Ladj.set(x, Ladj.zero(x))
    # hooks = [m.register_forward_hook(vmu.hooks.check_propagates_log_abs_det_jac)
    #         for m in model.backbone.concat.modules()]
    assert end is not None
    out = (model if end is None else vme.with_intermediate_outputs(model, end))(x)
    # for h in hooks:
    #    h.remove()
    return out


@dc.dataclass
class DiscriminativeFlowSupervisedTrainStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    bin_count: float = 256
    flow_end: str = None

    def __call__(self, trainer, batch):
        trainer.model.train()

        x, y = trainer.prepare_batch(batch)
        x = dequantize(x, 256)

        output, z = call_flow(trainer.model, x, end=self.flow_end)
        output, other_outputs = trainer.extend_output(output)

        loss_gen = vml.input_image_nll(x, z, self.bin_count).mean()
        loss_dis = trainer.loss(output, y).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen

        do_optimization_step(trainer.optimizer, loss)

        return NameDict(x=x, target=y, z=z, output=output, other_outputs=other_outputs,
                        loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.))


@dc.dataclass
class DiscriminativeFlowSupervisedEvalStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    bin_count: float = 256
    flow_end: str = None

    def __call__(self, trainer, batch):
        trainer.model.eval()

        x, y = trainer.prepare_batch(batch)

        output, z = call_flow(trainer.model, x, end=self.flow_end)
        output, other_outputs = trainer.extend_output(output)

        loss_gen = vml.input_image_nll(z, self.bin_count)
        loss_dis = trainer.loss(output, y).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen

        return NameDict(x=x, target=y, z=z, output=output, other_outputs=other_outputs,
                        loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.))


# Supervised multistep

@dc.dataclass
class SupervisedTrainMultiStep:
    """A training step where each batch is used multiple times consecutively.

    Args:
        repeat_count (int): Number of times each batch is repeated
            consecutively.
    """
    repeat_count: int

    def __call__(self, trainer, batch):
        trainer.model.train()
        for i in range(self.repeat_count):
            with batchnorm_stats_tracking_off(trainer.model) if i == 0 else ctx_suppress():
                x, y = trainer.prepare_batch(batch)
                output, other_outputs = trainer.extend_output(trainer.model(x))
                loss = trainer.loss(output, y).mean()
                do_optimization_step(trainer.optimizer, loss)
                if i == 0:
                    initial = dict(output=output, other_outputs=other_outputs, loss=loss.item())
        final = dict(output=output, other_outputs=other_outputs, loss=loss.item())
        return NameDict(x=x, target=y, **initial, **{f"{k}_post": v for k, v in final.items()})


@dc.dataclass
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
    steps_per_batch: int = None
    stride: int = None
    reversed: bool = True

    def __init__(self):
        if (self.stride is None) == (self.steps_per_batch is None):
            raise ValueError("Either `stride` or `steps_per_batch` should be provided.")
        self.prev_x_y, self.start_index = None, 0

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
            with batchnorm_stats_tracking_off(trainer.model) if i == starts[0] else ctx_suppress():
                inter_x, inter_y = [a[i:i + n] for a in (x, y)]
                output, other_outputs = trainer.extend_output(trainer.model(inter_x))
                loss = trainer.loss(output, inter_y).mean()
                do_optimization_step(trainer.optimizer, loss)
                result = result or NameDict(x=inter_x, target=inter_y, output=output,
                                            other_outputs=other_outputs, loss=loss.item())
        self.start_index = starts[0 if self.reversed else -1] + stride - len(self.prev_x_y[0])
        # Data for the last inter-batch iteration is returned
        # The last inter-batch is not necessarily `batch`
        # WARNING: Training performance might be too optimistic if reversed=False
        return result


# Supervised accumulated batch

@dc.dataclass
class SupervisedTrainAcumulatedBatchStep:
    batch_split_factor: int
    reduction: T.Literal['mean', 'sum'] = 'mean'

    def __post_init__(self):
        if self.reduction not in ['mean', 'sum']:
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
            loss = trainer.loss(output, y).mean()
            if self.reduction == 'mean':
                loss /= len(x)
            loss.backward()
            if total_loss is None:
                total_loss = loss.detach().clone()
            else:
                total_loss += loss.detach()
        trainer.optimizer.step()
        return NameDict(x=x, target=y, output=torch.cat(outputs, dim=0),
                        other_outputs=concatenate_tensors_trees(other_outputses),
                        loss=total_loss.item())


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


@dc.dataclass
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
    clean_proportion: float = 0
    virtual: bool = False

    def __call__(self, trainer, batch):
        """
        When clean_proportion = 0, the step reduces to
        >>> x, y = trainer.prepare_batch(batch)
        >>>
        >>> trainer.model.eval()
        >>> crc = CleanResultCallback(trainer.extend_output)
        >>> x_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y,
        >>>                                backward_callback=crc)
        >>> trainer.model.train()
        >>>
        >>> output, other_outputs = trainer.extend_output(trainer.model(x_p))
        >>> loss_p = trainer.loss(output, y).mean()
        >>> do_optimization_step(trainer.optimizer, loss=loss_p)
        >>>
        >>> return NameDict(x=x, output=crc.result.output, target=y,
        >>>                 other_outputs=crc.result.other_outputs, loss=crc.result.loss_mean,
        >>>                 x_p=x_p, target_p=y, output_p=output,
        >>>                 other_outputs_p=other_outputs, loss_p=loss_p.item())
        """
        x, y = trainer.prepare_batch(batch)
        cln_count = round(self.clean_proportion * len(x))
        cln_proportion = cln_count / len(x)
        split = lambda a: (a[:cln_count], a[cln_count:])
        (x_c, x_a), (y_c, y_a) = split(x), split(y)

        trainer.model.eval()  # adversarial examples are generated in eval mode
        crc = CleanResultCallback(trainer.extend_output)
        x_p = trainer.attack.perturb(trainer.model, x_a, None if self.virtual else y_a,
                                     backward_callback=crc)
        trainer.model.train()

        output, other_outputs = trainer.extend_output(trainer.model(torch.cat((x_c, x_p), dim=0)))
        output_c, output_p = split(output)
        loss_p = trainer.loss(output_p, y_a).mean()
        loss_c = trainer.loss(output_c, y_c).mean() if len(y_c) > 0 else 0
        do_optimization_step(trainer.optimizer,
                             loss=cln_proportion * loss_c + (1 - cln_proportion) * loss_p)

        other_outputs_p = NameDict({k: a[cln_count:] for k, a in other_outputs.items()})
        return NameDict(x=x, output=crc.result.output, target=y,
                        other_outputs=crc.result.other_outputs, loss=crc.result.loss_mean,
                        x_p=x_p, target_p=y_a, output_p=output_p,
                        other_outputs_p=other_outputs_p, loss_p=loss_p.item())


@dc.dataclass
class AdversarialCombinedLossTrainStep:
    """A training step that performs an optimization on an weighted
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
    use_attack_loss: bool = False
    clean_weight: float = 0.5
    adv_weight: float = None
    virtual: bool = False

    def __post_init__(self):
        if self.adv_weight is None:
            self.adv_weight = 1 - self.clean_weight

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)

        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y)

        trainer.model.train()
        output_c, other_outputs_c = trainer.extend_output(trainer.model(x))
        loss_c = trainer.loss(output_c, y).mean()
        output_p, other_outputs_p = trainer.extend_output(trainer.model(x_p))
        loss_p = (trainer.attack if self.use_attack_loss else trainer).loss(output_p, y).mean()

        do_optimization_step(trainer.optimizer,
                             loss=self.clean_weight * loss_c + self.adv_weight * loss_p)

        return NameDict(x=x, output=output_c, target=y, other_outputs=other_outputs_c,
                        loss=loss_c.item(),
                        x_p=x_p, output_p=output_p, other_outputs_p=other_outputs_p,
                        loss_p=loss_p.item())


@dc.dataclass
class AdversarialTrainBiStep:
    """A training step that first performs an optimization step on a clean batch
    and then on the batch turned into adversarial examples.

    Args:
        virtual (bool): A value determining whether virtual adversarial examples
            should be used (using the predicted label).
    """
    virtual: bool = False

    def __call__(self, trainer, batch):
        x, y = trainer.prepare_batch(batch)
        clean_result = _supervised_train_step_x_y(trainer, x, y)
        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y)
        result_p = _supervised_train_step_x_y(trainer, x_p, y)
        return NameDict(x=x, output=clean_result.output, target=y,
                        other_outputs=clean_result.other_outputs, loss=clean_result.loss_mean,
                        x_p=x_p, output_p=result_p.output,
                        other_outputs_p=result_p.other_outputs, loss_p=result_p.loss_mean)


@dc.dataclass
class AdversarialTrainMultiStep:
    """A training step that performs a model parameter update for each
    adversarial perturbation update.

    The perturbations of one batch can be used as initialization for the next
    one.
    """
    update_period: int = 1
    virtual: bool = False
    train_mode: bool = True
    reuse_pert: bool = False  # adversarial training for free
    last_pert: torch.Tensor = None

    def __call__(self, trainer, batch):
        # assert trainer.attack.loss == trainer.loss
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
        with batchnorm_stats_tracking_off(trainer.model) if self.train_mode else ctx_suppress():
            perturb = trainer.attack.perturb
            if self.reuse_pert:
                perturb = partial(perturb, initial_pert=self.last_pert)
            x_p = perturb(trainer.model, x, None if self.virtual else y, backward_callback=step)
            if self.reuse_pert:
                self.last_pert = x_p - x

        trainer.model.train()
        output_p, other_outputs_p = trainer.extend_output(trainer.model(x_p))
        loss_p = trainer.loss(output_p, y).mean()
        do_optimization_step(trainer.optimizer, loss_p)

        return NameDict(x=x, target=y, output=clean_result.output,
                        other_outputs=trainer.extend_output(clean_result.output)[1],
                        loss=clean_result.loss_mean, x_p=x_p, output_p=output_p,
                        other_outputs_p=other_outputs_p, loss_p=loss_p.item())


@dc.dataclass
class VATTrainStep:
    alpha: float = 1
    attack_eval_model: bool = False
    entropy_loss_coef: float = 0
    block_grad_for_clean: bool = True

    def __call__(self, trainer, batch):
        model, attack = trainer.model, trainer.attack
        model.train()

        x, y = trainer.prepare_batch(batch)
        output, other_outputs = trainer.extend_output(model(x))
        loss = trainer.loss(output, y).mean()
        with torch.no_grad() if self.block_grad_for_clean else ctx_suppress():
            target = attack.output_to_target(output)  # usually the same as other_outputs.probs
            if self.block_grad_for_clean:
                target = target.detach()
        with switch_training(model, False) if self.attack_eval_model else ctx_suppress():
            with batchnorm_stats_tracking_off(model) if model.training else ctx_suppress():
                x_p = attack.perturb(model, x, target)
                output_p, other_outputs_p = trainer.extend_output(model(x_p))
        loss_p = attack.loss(output_p, target).mean()
        loss = loss + self.alpha * loss_p
        loss_ent = vml.entropy_l(output_p).mean()
        if self.entropy_loss_coef:
            loss += self.entropy_loss_coef * loss_ent if self.entropy_loss_coef != 1 else loss_ent
        do_optimization_step(trainer.optimizer, loss=loss)

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_p=x_p, output_p=output_p, other_outputs_p=other_outputs_p,
                        loss_p=loss_p.item(), loss_ent=loss_ent.item())


@dc.dataclass
class PertConsistencyTrainStep:  # TODO
    alpha: float = 1
    entropy_loss_coef: float = 0
    track_pert_bn_stats: bool = False
    block_grad_for_clean: bool = True

    def __call__(self, trainer, batch):
        model = trainer.model
        model.train()

        x, y = trainer.prepare_batch(batch)
        output, other_outputs = trainer.extend_output(trainer.model(x))
        loss = trainer.loss(output, y).mean()
        with batchnorm_stats_tracking_off(model) if not self.track_pert_bn_stats \
                else ctx_suppress():
            with torch.no_grad():
                x_p, y_p = trainer.pert_model(x, y)
            output_p, other_outputs_p = trainer.extend_output(model(x_p))
            target_p = output.detach() if self.block_grad_for_clean else output
            loss_p = trainer.loss(output_p, target_p).mean()
            loss = loss + self.alpha * loss_p
            if self.entropy_loss_coef:
                loss_ent = vml.entropy_l(output_p).mean()
                loss += self.entropy_loss_coef * loss_ent
            do_optimization_step(trainer.optimizer, loss=loss)

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_p=x_p, target_p=target_p, output_p=output_p, other_outputs_p=other_outputs_p,
                        loss_p=loss_p.item(), loss_ent=loss_ent.item() if self.entropy_loss_coef else -1)


@torch.no_grad()
def _prepare_semisupervised_input(trainer, batch):
    x_u = None
    if isinstance(batch, BatchTuple):
        (x_l, y_l), (x_u, *_) = [trainer.prepare_batch(b) for b in batch]
    else:
        x_l, y_l = trainer.prepare_batch(batch)
    return x_l, y_l, x_u


def _get_unsupervised_vat_outputs(out, uns_start, block_grad_for_clean, output_to_target):
    out_uns = out[uns_start:]  # _get_unsupervised_vat_outputs(out, other_outs, uns_start)
    with torch.no_grad() if block_grad_for_clean else ctx_suppress():
        target_uns = output_to_target(out_uns)
        if block_grad_for_clean:
            target_uns = target_uns.detach()
    return out_uns, target_uns


@dc.dataclass
class SemisupVATEvalStep:
    consistency_loss_on_labeled: bool = True

    def __call__(self, trainer, batch):
        model, attack = trainer.model, trainer.attack
        model.eval()

        x_l, y_l, x_u = _prepare_semisupervised_input(trainer, batch)
        if x_u is None:
            x_c = x_all = x_l
            uns_start = 0
        else:
            x_all = torch.cat([x_l, x_u])
            x_c = x_all if self.consistency_loss_on_labeled else x_u
            uns_start = len(x_l) if self.consistency_loss_on_labeled else 0

        with torch.no_grad():
            out, other_outs = trainer.extend_output(model(x_all))
            out_uns, target_uns = _get_unsupervised_vat_outputs(
                out, uns_start, True, attack.output_to_target)

        x_p = attack.perturb(model, x_c, target_uns)

        with torch.no_grad():
            out_p, other_outs_p = trainer.extend_output(model(x_p))
            loss_p = attack.loss(out_p, attack.output_to_target(out_uns)).mean()
            loss_ent_p = vml.entropy_l(out_p).mean()

            out_l = out[:len(x_l)]
            loss_l = trainer.loss(out_l, y_l).mean()

        other_outs_l = type(other_outs)({k: v[:len(x_l)] for k, v in other_outs.items()})
        return NameDict(x=x_all, output=out, other_outputs=other_outs, output_l=out_l,
                        other_outputs_l=other_outs_l, loss_l=loss_l.item(), x_p=x_p,
                        loss_p=loss_p.item(), x_l=x_l, target=y_l, output_p=out_p,
                        other_outputs_p=other_outs_p, loss_ent_adv=loss_ent_p.item())


@dc.dataclass
class SemisupVATTrainStep:
    alpha: float = 1
    attack_eval_model: bool = False
    consistency_loss_on_labeled: bool = False
    entropy_loss_coef: float = 0
    block_grad_on_clean: bool = True

    def __call__(self, trainer, batch):
        model, attack = trainer.model, trainer.attack
        model.train()

        x_l, y_l, x_u = _prepare_semisupervised_input(trainer, batch)
        x = torch.cat([x_l, x_u])
        x_c, uns_start = (x, 0) if self.consistency_loss_on_labeled else (x_u, len(x_l))

        out, other_outs = trainer.extend_output(model(x))
        out_uns, target_uns = _get_unsupervised_vat_outputs(
            out, uns_start, self.block_grad_on_clean, attack.output_to_target)

        with switch_training(model, False) if self.attack_eval_model else ctx_suppress():
            with batchnorm_stats_tracking_off(model) if model.training else ctx_suppress():
                x_p = attack.perturb(model, x_c, target_uns)
                # NOTE: putting this outside of batchnorm_stats_tracking_off harms learning:
                out_p, other_outs_p = trainer.extend_output(model(x_p))

        loss_p = attack.loss(out_p, target_uns).mean()
        loss_l = trainer.loss(out_l := out[:len(x_l)], y_l).mean()
        loss = loss_l + self.alpha * loss_p
        loss_ent = vml.entropy_l(out_p).mean()
        if self.entropy_loss_coef:
            loss += self.entropy_loss_coef * loss_ent if self.entropy_loss_coef != 1 else loss_ent

        do_optimization_step(trainer.optimizer, loss=loss)

        other_outs_l = type(other_outs)({k: v[:len(x_l)] for k, v in other_outs.items()})

        return NameDict(x=x, output=out, other_outputs=other_outs, output_l=out_l,
                        other_outputs_l=other_outs_l, loss_l=loss_l.item(), loss_p=loss_p.item(),
                        x_l=x_l, target=y_l, x_p=x_p, output_p=out_p, other_outputs_p=other_outs_p,
                        loss_ent=loss_ent.item())


@dc.dataclass
class SemisupVATCorrEvalStep:
    alpha: float = 1
    attack_eval_model: bool = False
    entropy_loss_coef: float = 0
    block_grad_on_clean: bool = True
    corr_step_size: float = 1e-5

    def __call__(self, trainer, batch):
        model, attack = trainer.model, trainer.attack
        model.eval()

        x_l, y_l, x_u = _prepare_semisupervised_input(trainer, batch)
        assert x_u is None
        x = x_l
        x_c, uns_start = (x, 0)

        out, other_outs = trainer.extend_output(model(x))
        out_uns, target_uns = _get_unsupervised_vat_outputs(
            out, uns_start, self.block_grad_on_clean, attack.output_to_target)

        x_p = attack.perturb(model, x_c, target_uns)
        out_p, other_outs_p = trainer.extend_output(model(x_p))

        loss_p = attack.loss(out_p, target_uns).mean()
        loss_l = trainer.loss(out_l := out[:len(x_l)], y_l).mean()
        loss = loss_l + self.alpha * loss_p
        loss_ent = vml.entropy_l(out_p).mean()
        if self.entropy_loss_coef:
            loss += self.entropy_loss_coef * loss_ent if self.entropy_loss_coef != 1 else loss_ent

        import vidlu.torch_utils as vtu
        with vtu.save_params(model.parameters()):
            loss_p.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p += self.corr_step_size * p.grad
                    p.grad.zero_()

                out_c0 = out[uns_start:]  # clean logits before update (N×C)
                out_p0 = out_p  # perturbed logits before update
                out_c1 = model(x_c)
                out_p1 = model(x_p)

                dcp0 = out_c0 - out_p0
                dcp1 = out_c1 - out_p1
                uc = out_c1 - out_c0
                up = out_p1 - out_p0
                for x in [dcp0, dcp1, uc, up]:
                    x -= x.mean(dim=1, keepdims=True)
                    x /= x.norm(dim=1).view(-1, 1)

                dot = partial(torch.einsum, 'nc,nc->n')
                corr_c = dot(uc, dcp0).mean(0)
                corr_p = dot(up, dcp0).mean(0)
                corr_ucp = dot(uc, up).mean(0)
                corr_dcp = dot(dcp1, dcp0).mean(0)

        other_outs_l = type(other_outs)({k: v[:len(x_l)] for k, v in other_outs.items()})

        return NameDict(x=x, output=out, other_outputs=other_outs, output_l=out_l,
                        other_outputs_l=other_outs_l, loss_l=loss_l.item(), loss_p=loss_p.item(),
                        x_l=x_l, target=y_l, x_p=x_p, output_p=out_p, other_outputs_p=other_outs_p,
                        loss_ent=loss_ent.item(),
                        corr_c=corr_c.item(), corr_p=corr_p.item(), corr_ucp=corr_ucp.item(), corr_dcp=corr_dcp.item())


@dc.dataclass
class OldMeanTeacherTrainStep:  # TODO
    alpha: float = 1
    entropy_loss_coef: float = 0
    track_pert_bn_stats: bool = False
    block_grad_for_clean: bool = True

    def __call__(self, trainer, batch):
        m_student = trainer.model.student
        m_teacher = trainer.model.teacher  # teacher
        m_student.train()

        x, y = trainer.prepare_batch(batch)
        with torch.no_grad():
            output, other_outputs = trainer.extend_output(m_teacher(x))
        output, other_outputs = trainer.extend_output(trainer.m_student(x))
        loss = trainer.loss(output, y).mean()
        with batchnorm_stats_tracking_off(m_student) if not self.track_pert_bn_stats \
                else ctx_suppress():
            with torch.no_grad():
                x_p, y_p = trainer.pert_model(x, y)
            output_p, other_outputs_p = trainer.extend_output(m_student(x_p))
            target_p = output.detach() if self.block_grad_for_clean else output
            loss_p = trainer.loss(output_p, target_p).mean()
            loss = loss + self.alpha * loss_p
            if self.entropy_loss_coef:
                loss_ent = vml.entropy_l(output_p).mean()
                loss += self.entropy_loss_coef * loss_ent
            do_optimization_step(trainer.optimizer, loss=loss)

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_p=x_p, target_p=target_p, output_p=output_p, other_outputs_p=other_outputs_p,
                        loss_p=loss_p.item(), loss_ent=loss_ent.item() if self.entropy_loss_coef else -1)


@dc.dataclass
class MeanTeacherTrainStep:
    alpha: float = 1
    consistency_loss_on_labeled: bool = False  # True
    entropy_loss_coef: float = 0
    ema_teacher: nn.Module = None  # should this be here?
    ema_decay = 0.99
    clean_teacher_input: bool = False
    block_teacher_grad: bool = False

    def __call__(self, trainer, batch):
        assert not self.block_teacher_grad
        model, attack = trainer.model, trainer.attack
        if self.ema_teacher is None:
            self.ema_teacher = copy.deepcopy(model)
        elif next(self.ema_teacher.parameters()).device != next(model.parameters()).device:
            self.ema_teacher.to(model.device)
        teacher = self.ema_teacher
        model.train()
        teacher.train()

        x_l, y_l, x_u = _prepare_semisupervised_input(trainer, batch)
        x = torch.cat([x_l, x_u])
        x_c, uns_start = (x, 0) if self.consistency_loss_on_labeled else (x_u, len(x_l))

        out_l, other_outs_l = trainer.extend_output(model(x_l))
        # out_uns, target_uns = _get_unsupervised_vat_outputs(
        #    out, uns_start, self.block_grad_on_clean, attack.output_to_target)
        assert attack.output_to_target(out_l) is out_l
        x_pt = x_c if self.clean_teacher_input else attack.perturb(teacher, x_c)
        with torch.no_grad():
            out_pt, other_outs_pt = trainer.extend_output(teacher(x_pt))
        x_p = attack.perturb(model, x_c)
        out_p, other_outs_p = trainer.extend_output(model(x_p))

        loss_c = attack.loss(out_p, out_pt).mean()
        loss_l = trainer.loss(out_l, y_l).mean()
        loss = loss_l + self.alpha * loss_c
        loss_ent = vml.entropy_l(out_p).mean()
        if self.entropy_loss_coef:
            loss += self.entropy_loss_coef * loss_ent if self.entropy_loss_coef != 1 else loss_ent

        do_optimization_step(trainer.optimizer, loss=loss)
        with torch.no_grad():
            for p, pt in zip(model.parameters(), teacher.parameters()):
                pt.mul_(self.ema_decay).add(p, alpha=1 - self.ema_decay)

        return NameDict(x=x, output=None, other_outputs=None, output_l=out_l,
                        other_outputs_l=other_outs_l, loss_l=loss_l.item(), loss_c=loss_c.item(),
                        x_l=x_l, target=y_l, x_p=x_p, output_p=out_p, other_outputs_p=other_outs_p,
                        loss_ent=loss_ent.item())

    def state_dict(self):
        return self.ema_teacher.state_dict()

    def load_state_dict(self, state_dict):  # TODO
        self.ema_teacher.load_state_dict(state_dict)


@dc.dataclass
class AdversarialEvalStep:
    virtual: bool = False
    entropy_loss_coef: float = 0

    def __call__(self, trainer, batch):
        trainer.model.eval()
        attack = trainer.eval_attack

        x, y = trainer.prepare_batch(batch)

        with torch.no_grad():
            output, other_outputs = trainer.extend_output(trainer.model(x))
            loss = trainer.loss(output, y).mean()

        x_p = attack.perturb(trainer.model, x, None if self.virtual else y)
        with torch.no_grad():
            output_p, other_outputs_p = trainer.extend_output(trainer.model(x_p))
            loss_p = (attack.loss(output_p, attack.output_to_target(output)) if self.virtual
                      else trainer.loss(output_p, y)).mean()

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_p=x_p, output_p=output_p, other_outputs_p=other_outputs_p,
                        loss_p=loss_p.item())


class AdversarialTargetedEvalStep:
    def __init__(self, targets=None, class_count=None):
        if (targets is None) == (class_count is None):
            raise RuntimeError("Either the `targets` or the `class_count`"
                               + " argument needs to be provided.")
        self.targets = list(range(class_count)) if targets is None else targets

    def __call__(self, trainer, batch):
        trainer.model.eval()

        x, y = trainer.prepare_batch(batch)

        with torch.no_grad():
            output, other_outputs = trainer.extend_output(trainer.model(x))
            loss = trainer.loss(output, y).mean()

        result = dict()
        for i, t in enumerate(self.targets):
            t_var = torch.full_like(y, t)
            x_p = trainer.eval_attack.perturb(trainer.model, x, t_var)
            with torch.no_grad():
                result[f"output_p{i}"], result[f"other_outputs_p{i}"] = trainer.extend_output(
                    trainer.model(x_p))
                result[f"loss_p_targ{i}"] = trainer.loss(result[f"output_p{i}"],
                                                         t_var).mean().item()
                result[f"loss_p{i}"] = trainer.loss(result[f"output_p{i}"], y).mean().item()

        return NameDict(x=x, target=y, output=output, other_outputs=other_outputs, loss=loss.item(),
                        x_p=x_p, **result)


# Autoencoder

def autoencoder_train_step(trainer, batch):
    trainer.model.train()
    x = trainer.prepare_batch(batch)[0]
    x_r = trainer.model(x)
    loss = trainer.loss(x_r, x).mean()
    do_optimization_step(trainer.optimizer, loss)
    return NameDict(x_r=x_r, x=x, loss=loss.item())


'''
# GAN

class GANTrainStep:
    @lru_cache(1)
    def _get_real_labels(self, batch_size):
        return torch.ones(batch_size, device=self.model.device)

    @lru_cache(1)
    def _get_fake_labels(self, batch_size):
        return torch.zeros(batch_size, device=self.model.device)

    def __call__(self, trainer, batch):
        """ Copied from ignite/examples/gan/dcgan and modified"""
        trainer.model.train()
        real = trainer.prepare_batch(batch)[0]
        batch_size = real.shape[0]
        real_labels = self._get_real_labels(batch_size)

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) ##########
        discriminator, generator = trainer.model.discriminator, trainer.model.generator
        discriminator.zero_grad()

        # training discriminator with real
        output = discriminator(real)
        errD_real = trainer.loss(output, real_labels).mean()  # torch.nn.BCELoss()
        D_real = output.mean().item()
        errD_real.backward()

        fake = generator(trainer.model.sample_z(batch_size))

        # training discriminator with fake
        output = discriminator(fake.detach())
        errD_fake = trainer.loss(output, self._get_fake_labels(batch_size)).mean()
        D_fake1 = output.mean().item()
        errD_fake.backward()

        trainer.optimizer['D'].step()

        # (2) Update G network: maximize log(D(G(z))) ##########################
        generator.zero_grad()

        # Update generator.
        # We want to make a step that will make it more likely that D outputs "real"
        output = discriminator(fake)
        errG = trainer.loss(output, real_labels).mean()
        D_fake2 = output.mean().item()

        errG.backward()
        trainer.optimizer['G'].step()

        return NameDict(errD=(errD_real + errD_fake).item(), errG=errG.item(), D_real=D_real,
                        D_fake1=D_fake1, D_fake2=D_fake2)
'''
