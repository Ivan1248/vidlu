import contextlib as ctx
import dataclasses as dc
from vidlu.utils.func import partial
import typing as T
from torch import nn
import copy
from warnings import warn

import numpy as np
import torch
from torch import nn

from vidlu.data import BatchTuple, Record
from vidlu.utils.collections import NameDict
from vidlu.torch_utils import (switch_training, batchnorm_stats_tracking_off)
import vidlu.modules as vm
import vidlu.modules.losses as vml
import vidlu.modules.utils as vmu

from vidlu.modules.tensor_extra import LogAbsDetJac as Ladj


# Training/evaluation steps ########################################################################

@dc.dataclass
class TrainerStep:  # TODO: use to reduce
    outputs: dict = dc.field(
        default_factory=lambda: dict(
            zip(*[["x", "target", "out", "loss"]] * 2)))

    def __call__(self, trainer, batch):
        output = self.run(trainer, batch)
        return NameDict({self.outputs[k]: v for k, v in output.items() if k in self.outputs})


@dc.dataclass
class BaseStep:
    metric_fs = []
    out_names = ['loss']

    def __call__(self, trainer, batch):
        result = self.run(trainer, batch)
        return {k: result['k'] for k in self.out_names}


# Supervised
@torch.no_grad()
def _prepare_sup_batch(batch):
    """Extracts labeled and unlabeled batches.

    Args:
        batch: See the code.

    Returns:
        x - inputs, y - labels
    """
    if isinstance(batch, BatchTuple):
        warn("The batch (batchTuple) consists of 2 batches.")
        x, y = tuple(torch.cat(a, 0) for a in zip(*batch))
    else:
        x, y = batch
    return x, y


def do_optimization_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@ctx.contextmanager
def optimization_step(optimizer):
    optimizer.zero_grad()
    yield
    optimizer.step()


@torch.no_grad()
def supervised_eval_step(trainer, batch):
    trainer.model.eval()
    x, y = _prepare_sup_batch(batch)
    out = trainer.model(x)
    loss = trainer.loss(out, y).mean()
    return NameDict(x=x, target=y, out=out, loss=loss.item())


def _supervised_train_step_x_y(trainer, x, y):
    trainer.model.train()
    out = trainer.model(x)
    loss = trainer.loss(out, y).mean()
    do_optimization_step(trainer.optimizer, loss)
    return NameDict(x=x, target=y, out=out, loss=loss.item())


def supervised_train_step(trainer, batch):
    x, y = _prepare_sup_batch(batch)
    return _supervised_train_step_x_y(trainer, x, y)


@dc.dataclass
class ClassifierEnsembleEvalStep:
    model_iter: T.Callable[[nn.Module], T.Iterator[nn.Module]]
    combine: T.Callable[[T.Sequence], torch.Tensor] = lambda x: torch.stack(x).mean(0)

    def __call__(self, trainer, batch):
        trainer.model.eval()
        x, y = batch
        out = self.combine(model(x) for model in self.model_iter(trainer.model))
        loss = trainer.loss(out, y).mean()
        return NameDict(x=x, target=y, out=out, loss=loss.item())


# generative flows and discriminative hybrids

def dequantize(x, bin_count, scale=1):
    return x + scale / bin_count * torch.rand_like(x)


def call_flow(model, x, end: T.Union[str, T.Tuple[str, slice]] = None):
    # hooks = [m.register_forward_hook(vmu.hooks.check_propagates_log_abs_det_jac)
    #         for m in model.backbone.concat.modules()]
    assert end is not None
    end_name, z_slice = (end, None) if isinstance(end, str) else end
    assert z_slice is None
    Ladj.set(x, Ladj.zero(x))
    out, z = (model if end is None else vm.with_intermediate_outputs(model, end_name))(x)
    # if slice is not None:
    #    z = z[z_slice]
    # for h in hooks:
    #    h.remove()
    return out, z


@dc.dataclass
class DiscriminativeFlowSupervisedTrainStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    bin_count: float = 256
    flow_end: T.Union[str, T.Tuple[str, slice]] = None

    def __call__(self, trainer, batch):
        trainer.model.train()
        x, y = batch

        x = dequantize(x, 256)
        out, z = call_flow(trainer.model, x, end=self.flow_end)

        if not isinstance(z, torch.Tensor):
            z = torch.cat([zi.view(x.shape[0], -1) for zi in vmu.extract_tensors(z)])

        loss_gen = vml.input_image_nll(x, z, self.bin_count).mean()
        loss_dis = trainer.loss(out, y).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen

        do_optimization_step(trainer.optimizer, loss)

        return NameDict(x=x, target=y, z=z, out=out, loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.))


@dc.dataclass
class DiscriminativeFlowSupervisedEvalStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    bin_count: float = 256
    flow_end: str = None

    @torch.no_grad()
    def __call__(self, trainer, batch):
        trainer.model.eval()
        x, y = batch

        out, z = call_flow(trainer.model, x, end=self.flow_end)

        loss_gen = vml.input_image_nll(x, z, self.bin_count).mean()
        loss_dis = trainer.loss(out, y).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen

        return NameDict(x=x, target=y, z=z, out=out, loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.))


@dc.dataclass
class AdversarialDiscriminativeFlowSupervisedTrainStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    adv_weight: float = 1.
    bin_count: float = 256
    flow_end: T.Union[str, T.Tuple[str, slice]] = None

    def _adv_loss(self, out_adv):
        return -out_adv.sigmoid().log()

    def __call__(self, trainer, batch):
        d_temp = 1
        trainer.model.train()
        x, y = batch

        x = dequantize(x, 256)

        out_full, z = call_flow(trainer.model, x, end=self.flow_end)
        out, out_d = out_full[0][:, :-1], out_full[0][:, -1:]
        if not isinstance(z, torch.Tensor):
            z_ = Ladj.add(torch.cat(z, dim=1), z, Ladj.zero(z[0]))

        loss_gen = vml.input_image_nll(x, z_, self.bin_count).mean()
        loss_dis = trainer.loss(out, y).mean()
        loss_adv = self._adv_loss(out_d / d_temp).mean()
        if loss_adv_inf := torch.isinf(loss_adv):
            print(f"loss_adv={loss_adv.item()}, {loss_adv=}")
            loss_adv = torch.full_like(loss_adv, 10000.)
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen + self.adv_weight * loss_adv

        do_optimization_step(trainer.optimizer, loss)

        # adversarial step

        if self.adv_weight != 0 and not loss_adv_inf:
            with batchnorm_stats_tracking_off(trainer.model):
                with torch.no_grad():
                    out_adv = torch.roll(out, 1, dims=0)
                    out_full_adv = (torch.cat([out_adv, out_d], dim=1).detach(),
                                    *[x.detach() for x in out_full[1:]])
                    x_adv = trainer.model.inverse(out_full_adv)
                    x_adv_rg = x_adv  # vm.rev_grad(x_adv)
                out_adv_, z = call_flow(trainer.model, x_adv_rg, end=self.flow_end)
            out_adv_cd, out_adv_d = out_adv_[0][:, :-1], out_adv_[0][:, -1:]

            loss_adv_ = self._adv_loss(-out_adv_d / d_temp).mean()

            if torch.isinf(loss_adv_):
                print(f"loss_adv_={loss_adv_.item()}")
            else:
                do_optimization_step(trainer.optimizer, self.adv_weight * loss_adv_)
        else:
            loss_adv_ = loss_adv * 0 - 1

        # for h in hooks:
        #     h.remove()

        return NameDict(x=x, target=y, z=z, out=out, loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.), loss_adv=loss_adv.item(),
                        loss_adv_=loss_adv_.item())


@dc.dataclass
class AdversarialDiscriminativeFlowSupervisedEvalStep:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    adv_weight: float = 1.
    bin_count: float = 256
    flow_end: T.Union[str, T.Tuple[str, slice]] = None

    def _adv_loss(self, out_adv):
        return -out_adv.sigmoid().log()

    @torch.no_grad()
    def __call__(self, trainer, batch):
        trainer.model.eval()
        x, y = batch

        x = dequantize(x, 256)
        out_cd, z = call_flow(trainer.model, x, end=self.flow_end)
        out, out_d = out_cd[0][:, :-1], out_cd[0][:, -1:]
        if not isinstance(z, torch.Tensor):
            z_ = Ladj.add(torch.cat(z, dim=1), z, Ladj.zero(z[0]))

        loss_gen = vml.input_image_nll(x, z_, self.bin_count).mean()
        loss_dis = trainer.loss(out, y).mean()
        loss_adv = self._adv_loss(out_d).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen + self.adv_weight * loss_adv

        return NameDict(x=x, target=y, z=z, out=out, loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.), loss_adv=loss_adv.item())


@dc.dataclass
class AdversarialDiscriminativeFlowSupervisedTrainStep2:  # TODO: improve
    gen_weight: float = 1.  # bit/dim
    dis_weight: float = 1.
    adv_weight: float = 1.
    bin_count: float = 256
    flow_end: T.Union[str, T.Tuple[str, slice]] = None
    eval: bool = False

    def _adv_loss(self, out_adv):
        return -out_adv.sigmoid().log()

    def __call__(self, trainer, batch):
        trainer.model.train()
        x, y = batch

        x = dequantize(x, 256)
        out_full, z = call_flow(trainer.model, x, end=self.flow_end)
        logits = out_full[0]
        if not isinstance(z, torch.Tensor):
            z_ = Ladj.add(torch.cat(z, dim=1), z, Ladj.zero(z[0]))

        loss_gen = vml.input_image_nll(x, z_, self.bin_count).mean()
        loss_dis = trainer.loss(logits, y).mean()
        loss = self.dis_weight * loss_dis + self.gen_weight * loss_gen

        if not self.eval:
            do_optimization_step(trainer.optimizer, loss)

        # adversarial step

        with batchnorm_stats_tracking_off(trainer.model):
            with torch.no_grad():
                logits_adv = torch.roll(logits, 1, dims=0)
                out_full_adv = (logits_adv.detach(),
                                *[x.detach() for x in out_full[1:]])
                x_adv = trainer.model.inverse(out_full_adv)
            out_full_adv_, z = call_flow(trainer.model, x_adv.clamp(x.min(), x.max()),
                                         end=self.flow_end)
            logits_adv_ = out_full_adv_[0]

        loss_adv = trainer.loss(logits_adv_, y).mean()

        if not self.eval:
            do_optimization_step(trainer.optimizer, self.adv_weight * loss_adv)

        return NameDict(x=x, target=y, z=z, out=logits, loss=loss.item(), loss_dis=loss_dis.item(),
                        loss_gen_b=loss_gen.item() / np.log(2.), loss_adv=loss_adv.item())


@dc.dataclass
class AdversarialDiscriminativeFlowSupervisedEvalStep2(
    AdversarialDiscriminativeFlowSupervisedTrainStep2):
    eval: bool = True

    __call__ = torch.no_grad()(AdversarialDiscriminativeFlowSupervisedTrainStep2.__call__)


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
        x, y = batch
        for i in range(self.repeat_count):
            with batchnorm_stats_tracking_off(trainer.model) if i == 0 else ctx.suppress():
                out = trainer.model(x)
                loss = trainer.loss(out, y).mean()
                do_optimization_step(trainer.optimizer, loss)
                if i == 0:
                    initial = dict(out=out, loss=loss.item())
        final = dict(out=out, loss=loss.item())
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
        x_y = batch
        n = len(x_y[0])
        stride = self.stride or n // self.steps_per_batch

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
            with batchnorm_stats_tracking_off(trainer.model) if i == starts[0] else ctx.suppress():
                inter_x, inter_y = [a[i:i + n] for a in (x, y)]
                out = trainer.model(inter_x)
                loss = trainer.loss(out, inter_y).mean()
                do_optimization_step(trainer.optimizer, loss)
                result = result or NameDict(x=inter_x, target=inter_y, out=out, loss=loss.item())
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
        x, y = batch
        trainer.optimizer.zero_grad()
        xs, ys = [b.split(len(x) // self.batch_split_factor, dim=0) for b in [x, y]]
        outputs = []
        total_loss = None
        for x, y in zip(xs, ys):
            out = trainer.model(x)
            outputs.append(out)
            loss = trainer.loss(out, y).mean()
            if self.reduction == 'mean':
                loss /= len(x)
            loss.backward()
            if total_loss is None:
                total_loss = loss.detach().clone()
            else:
                total_loss += loss.detach()
        trainer.optimizer.step()
        return NameDict(x=x, target=y, out=torch.cat(outputs, dim=0), loss=total_loss.item())


# Adversarial

class CleanResultCallback:
    def __init__(self):
        self.result = None

    def __call__(self, r):
        if self.result is None:
            self.result = NameDict(vars(r), out=r.out)


@dc.dataclass
class AdversarialTrainStep:
    """Adversarial training step with the option to keep a the batch partially
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
        >>> x, y = batch
        >>>
        >>> trainer.model.eval()
        >>> crc = CleanResultCallback()
        >>> x_p, y_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y,
        >>>                                   backward_callback=crc)
        >>> trainer.model.train()
        >>>
        >>> out = trainer.model(x_p)
        >>> loss_p = trainer.loss(out, y).mean()
        >>> do_optimization_step(trainer.optimizer, loss=loss_p)
        """
        x, y = batch
        cln_count = round(self.clean_proportion * len(x))
        cln_proportion = cln_count / len(x)
        split = lambda a: (a[:cln_count], a[cln_count:])
        (x_c, x_a), (y_c, y_a) = split(x), split(y)

        trainer.model.eval()  # adversarial examples are generated in eval mode
        crc = CleanResultCallback()
        x_p, y_p = trainer.attack.perturb(trainer.model, x_a, None if self.virtual else y_a,
                                          backward_callback=crc)
        trainer.model.train()

        out = trainer.model(torch.cat((x_c, x_p), dim=0))
        out_c, out_p = split(out)
        loss_p = trainer.loss(out_p, y_p).mean()
        loss_c = trainer.loss(out_c, y_c).mean() if len(y_c) > 0 else 0
        do_optimization_step(trainer.optimizer,
                             loss=cln_proportion * loss_c + (1 - cln_proportion) * loss_p)
        return NameDict(x=x, out=crc.result.out, target=y, loss=crc.result.loss_mean,
                        x_p=x_p, target_p=y_p, out_p=out_p, loss_p=loss_p.item())


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
        x, y = batch

        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_p, y_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y)

        trainer.model.train()
        out_c = trainer.model(x)
        loss_c = trainer.loss(out_c, y_p).mean()
        out_p = trainer.model(x_p)
        loss_p = (trainer.attack if self.use_attack_loss else trainer).loss(out_p, y).mean()

        do_optimization_step(trainer.optimizer,
                             loss=self.clean_weight * loss_c + self.adv_weight * loss_p)

        return NameDict(x=x, out=out_c, target=y, loss=loss_c.item(), x_p=x_p, out_p=out_p,
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
        x, y = batch
        clean_result = _supervised_train_step_x_y(trainer, x, y)
        trainer.model.eval()  # adversarial examples are generated in eval mode
        x_p, y_p = trainer.attack.perturb(trainer.model, x, None if self.virtual else y)
        result_p = _supervised_train_step_x_y(trainer, x_p, y_p)
        return NameDict(x=x, out=clean_result.out, target=y, loss=clean_result.loss_mean,
                        x_p=x_p, out_p=result_p.out, loss_p=result_p.loss_mean)


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
        x, y = batch
        clean_result = None

        def step(r):
            nonlocal clean_result, x, self
            if clean_result is None:
                clean_result = r
            if r.step % self.update_period == 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

        (trainer.model.train if self.train_mode else trainer.model.eval)()
        with batchnorm_stats_tracking_off(trainer.model) if self.train_mode else ctx.suppress():
            perturb = trainer.attack.perturb
            if self.reuse_pert:
                perturb = partial(perturb, initial_pert=self.last_pert)
            x_p, y_p = perturb(trainer.model, x, None if self.virtual else y,
                               backward_callback=step)
            if self.reuse_pert:
                self.last_pert = x_p - x

        trainer.model.train()
        out_p = trainer.model(x_p)
        loss_p = trainer.loss(out_p, y_p).mean()
        do_optimization_step(trainer.optimizer, loss_p)

        return NameDict(x=x, target=y, out=clean_result.out, loss=clean_result.loss_mean,
                        x_p=x_p, out_p=out_p, loss_p=loss_p.item())


@dc.dataclass
class VATTrainStep:
    alpha: float = 1
    attack_eval_model: bool = False
    entropy_loss_coef: float = 0
    block_grad_for_clean: bool = True

    def __call__(self, trainer, batch):
        model, attack = trainer.model, trainer.attack
        model.train()
        x, y = batch

        out = model(x)
        loss = trainer.loss(out, y).mean()
        with torch.no_grad() if self.block_grad_for_clean else ctx.suppress():
            target = attack.output_to_target(out)  # usually the same as target.sogtmax(1)
            if self.block_grad_for_clean:
                target = target.detach()
        with switch_training(model, False) if self.attack_eval_model else ctx.suppress():
            with batchnorm_stats_tracking_off(model) if model.training else ctx.suppress():
                x_p, y_p = attack.perturb(model, x, target)
                out_p = model(x_p)
        loss_p = attack.loss(out_p, y_p).mean()
        loss = loss + self.alpha * loss_p
        loss_ent = vml.entropy_l(out_p).mean()
        if self.entropy_loss_coef:
            loss += self.entropy_loss_coef * loss_ent if self.entropy_loss_coef != 1 else loss_ent
        do_optimization_step(trainer.optimizer, loss=loss)

        return NameDict(x=x, target=y, out=out, loss=loss.item(), x_p=x_p, out_p=out_p,
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
        x, y = batch

        out = model(x)
        loss = trainer.loss(out, y).mean()
        with batchnorm_stats_tracking_off(model) if not self.track_pert_bn_stats \
                else ctx.suppress():
            with torch.no_grad():
                x_p, y_p = trainer.pert_model(x, y)
            out_p = model(x_p)
            target_p = out.detach() if self.block_grad_for_clean else out
            loss_p = trainer.loss(out_p, target_p).mean()
            loss = loss + self.alpha * loss_p
            if self.entropy_loss_coef:
                loss_ent = vml.entropy_l(out_p).mean()
                loss += self.entropy_loss_coef * loss_ent
            do_optimization_step(trainer.optimizer, loss=loss)

        return NameDict(x=x, target=y, out=out, loss=loss.item(), x_p=x_p, target_p=target_p,
                        out_p=out_p, loss_p=loss_p.item(),
                        loss_ent=loss_ent.item() if self.entropy_loss_coef else -1)


@torch.no_grad()
def _prepare_semisup_batch(batch):
    """Extracts labeled and unlabeled batches.

    Args:
        batch: See the code.

    Returns:
        x_l - labeled inputs, y_l - labeled labels, x_u - unlabeled inputs
    """
    x_u = None
    if isinstance(batch, BatchTuple):
        (x_l, y_l), (x_u, *_) = batch
    elif isinstance(batch, Record) and 'x_u' in batch:
        x_l, x_u, y_l = batch.x_l, batch.x_u, batch.y_l
    else:
        x_l, y_l = batch
    return x_l, y_l, x_u


def _prepare_semisup_batch_s(batch, unsup_loss_on_all=False):
    """Returns arrays representing supervised inputs `x_l`, labels `y_l`,
    unsupervised inputs `x_u`, and all inputs `x_all`.

    `x_u` includes with `x_l` at the beginning if `unsup_loss_on_all=True`.

    Args:
        batch: A data structure that `_prepare_semisup_batch` accepts.
        unsup_loss_on_all: Whether the unsupervised batch `x_u` should include
            the supervised inputs `x_l`.
    """
    x_l, y_l, x_u = _prepare_semisup_batch(batch)
    x_all = x_l if x_u is None else torch.cat([x_l, x_u])
    x_u = x_all if unsup_loss_on_all else x_u

    if x_u is None and not unsup_loss_on_all:
        raise RuntimeError(f"uns_loss_on_all must be True if there is no unlabeled batch.")
    return x_l, y_l, x_u, x_all


def _cons_output_to_target(out_uns, block_grad, output_to_target):
    with torch.no_grad() if block_grad else ctx.suppress():
        target_uns = output_to_target(out_uns)
    if block_grad:  # for the case when output_to_target is identity
        target_uns = target_uns.detach()
    return target_uns


# TODO: check TODO
def _perturb(attack, model, x, attack_target, loss_mask='create', attack_eval_model=False,
             bn_stats_updating=False, detach_out=False):
    """Applies corresponding perturbations to the input, unsupervised target, and validity mask.
    Also computes the prediction in the perturbed input."""
    if loss_mask == 'create':
        loss_mask = torch.ones_like(attack_target[:, 0, ...])

    with switch_training(model, False) if attack_eval_model else \
            batchnorm_stats_tracking_off(model) if model.training and not bn_stats_updating else \
                    ctx.suppress():
        pmodel = attack(model, x, attack_target, loss_mask=loss_mask)
        x_p, target_p, loss_mask_p = pmodel(x, attack_target, loss_mask)
        with torch.no_grad() if detach_out else ctx.suppress():  # TODO: enable enabling batchnorm stats tracking here, CHECK!
            out_p = model(x_p)
            if detach_out:
                out_p = out_p.detach()
    return NameDict(x=x_p, target=target_p, loss_mask=loss_mask_p, out=out_p)


@dc.dataclass
class SemisupCleanTargetConsStepBase:
    """Base class for VAT, mean teacher, ..."""
    alpha: float = 1
    attack_eval_model: bool = False
    pert_bn_stats_updating: bool = False
    uns_loss_on_all: bool = False
    entropy_loss_coef: float = 0
    loss_cons: T.Optional[T.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    block_grad_on_clean: bool = True
    block_grad_on_pert: bool = False
    pert_both: bool = False
    rev_cons: bool = False
    mem_efficient: bool = True
    eval: bool = False

    def get_student_and_teacher(self, trainer):
        return [trainer.model] * 2

    def __call__(self, trainer, batch):
        x_l, y_l, x_u, x_all = _prepare_semisup_batch_s(batch, self.uns_loss_on_all)
        detach_clean = self.eval or self.block_grad_on_clean
        detach_pert = self.eval or self.block_grad_on_pert

        attack = trainer.attack
        model, teacher = self.get_student_and_teacher(trainer)
        lc = self.loss_cons or attack.loss
        loss_cons = (lambda a, b: lc(b, a)) if self.rev_cons else lc
        output_to_target = partial(_cons_output_to_target, block_grad=self.block_grad_on_clean,
                                   output_to_target=attack.output_to_target)
        mem_eff = teacher is not model or self.mem_efficient and not self.uns_loss_on_all
        perturb_input = lambda attack_target, detach_out: _perturb(
            attack=attack, model=model, x=x_u, attack_target=attack_target, detach_out=detach_out,
            attack_eval_model=self.attack_eval_model, bn_stats_updating=self.pert_bn_stats_updating)

        model.eval() if self.eval else model.train()  # teacher always eval. mode if it's not model

        with optimization_step(trainer.optimizer) if not self.eval else ctx.suppress():
            with torch.no_grad() if self.eval else ctx.suppress():
                out_l = (out_all := model(x_all))[:len(y_l)] if not mem_eff else model(x_l)
                loss_l = trainer.loss(out_l, y_l).mean()
                if not self.eval and mem_eff:  # back-propagates supervised loss sooner
                    loss_l.backward()
                    loss_l = loss_l.detach()

            if self.alpha == 0:
                return NameDict(x=x_l, target=y_l, out=out_l, loss_l=loss_l.item(),
                                loss_u=0, x_u=x_u, x_p=None, y_p=None, out_u=None, out_p=None,
                                loss_ent=0)

            with torch.no_grad() if detach_clean else ctx.suppress():
                if self.pert_both:
                    out_u = perturb_input(attack_target=x_u.new_zeros(x_u[:, :1].shape),
                                          detach_out=detach_clean).out
                else:  # default
                    out_u = teacher(x_u) if mem_eff else out_all[-len(x_u):]
                target_uns = output_to_target(out_u)  # usually identity with .detach()
            pert = perturb_input(attack_target=target_uns, detach_out=detach_pert)

            with torch.no_grad() if self.eval else ctx.suppress():
                if self.block_grad_on_pert:  # non-default
                    pert.out = pert.out.detach()
                if self.block_grad_on_clean:  # just in case
                    pert.target = pert.target.detach()
                loss_u = loss_cons(pert.out, pert.target)[pert.loss_mask >= 1 - 1e-6].mean()
                loss = loss_l + self.alpha * loss_u
                loss_ent = vml.entropy_l(pert.out).mean()
                if self.entropy_loss_coef:
                    loss += self.entropy_loss_coef * loss_ent
            if not self.eval:
                loss.backward()

        return NameDict(x=x_l, target=y_l, out=out_l, loss_l=loss_l.item(), loss_u=loss_u.item(),
                        x_u=x_u, x_p=pert.x, y_p=pert.target, out_u=out_u, out_p=pert.out,
                        loss_ent=loss_ent.item())


@dc.dataclass
class SemisupVATStep(SemisupCleanTargetConsStepBase):
    pass


SemisupVATTrainStep = SemisupVATStep  # TODO: delete


@dc.dataclass
class SemisupVATEvalStep(SemisupVATStep):
    eval: bool = True
    uns_loss_on_all: bool = True


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

        x_l, y_l, x_u = _prepare_semisup_batch(batch)
        assert x_u is None
        x = x_l
        x_c, uns_start = (x, 0)

        out = model(x)
        out_uns, target_uns = _cons_output_to_target(
            out, uns_start, self.block_grad_on_clean, attack.output_to_target)

        x_p, y_p = attack.perturb(model, x_c, target_uns)
        out_p = model(x_p)

        loss_p = attack.loss(out_p, y_p).mean()
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

        return NameDict(x=x, out=out, out_l=out_l, loss_l=loss_l.item(), loss_p=loss_p.item(),
                        x_l=x_l, target=y_l, x_p=x_p, out_p=out_p, loss_ent=loss_ent.item(),
                        corr_c=corr_c.item(), corr_p=corr_p.item(), corr_ucp=corr_ucp.item(),
                        corr_dcp=corr_dcp.item())


@torch.no_grad()
def update_mean_teacher(teacher, model, ema_decay):
    # Note that not only optimized parameters are updated.
    model_dict, teacher_dict = model.state_dict(), teacher.state_dict()
    if len(model_dict) != len(teacher_dict):
        raise RuntimeError("State dicts of the models are different.")
    for k, p in model_dict.items():
        if torch.is_floating_point(p):
            teacher_dict[k].mul_(ema_decay).add_(p, alpha=1 - ema_decay)


@dc.dataclass
class MeanTeacherStep(SemisupCleanTargetConsStepBase):
    ema_decay: float = 0.99
    ema_teacher: T.Optional[T.Union[nn.Module, dict]] = None  # should this be here?

    def __call__(self, trainer, batch):
        result = super().__call__(trainer, batch)
        update_mean_teacher(self.ema_teacher, trainer.model, self.ema_decay)
        return result

    def get_student_and_teacher(self, trainer):
        model = trainer.model
        if self.ema_teacher is None or isinstance(self.ema_teacher, dict):
            state_dict, self.ema_teacher = self.ema_teacher, copy.deepcopy(model)
            self.ema_teacher.eval()
            if state_dict is not None:
                self.ema_teacher.load_state_dict(state_dict)
        elif next(self.ema_teacher.parameters()).device != next(model.parameters()).device:
            self.ema_teacher.to(model.device)
        return trainer.model, self.ema_teacher

    def state_dict(self):
        t = self.ema_teacher
        return t if isinstance(t, dict) else dict() if isinstance(t, dict) else t.state_dict()

    def load_state_dict(self, state_dict):
        if self.ema_teacher is None or isinstance(self.ema_teacher, dict):
            self.ema_teacher = dict(state_dict) if len(state_dict) > 0 else None
        else:
            self.ema_teacher.load_state_dict(state_dict)


@dc.dataclass
class AdversarialEvalStep:
    virtual: bool = False
    entropy_loss_coef: float = 0

    def __call__(self, trainer, batch):
        trainer.model.eval()
        attack = trainer.eval_attack
        x, y = batch

        with torch.no_grad():
            out = trainer.model(x)
            loss = trainer.loss(out, y).mean()

        x_p, y_p = attack.perturb(trainer.model, x, None if self.virtual else y)
        with torch.no_grad():
            out_p = trainer.model(x_p)
            loss_p = (attack.loss(out_p, attack.output_to_target(out)) if self.virtual
                      else trainer.loss(out_p, y)).mean()

        return NameDict(x=x, target=y, out=out, loss=loss.item(), x_p=x_p, out_p=out_p,
                        loss_p=loss_p.item())


class AdversarialTargetedEvalStep:
    def __init__(self, targets=None, class_count=None):
        if (targets is None) == (class_count is None):
            raise RuntimeError("Either the `targets` or the `class_count`"
                               + " argument needs to be provided.")
        self.targets = list(range(class_count)) if targets is None else targets

    def __call__(self, trainer, batch):
        trainer.model.eval()
        x, y = batch

        with torch.no_grad():
            output = trainer.model(x)
            loss = trainer.loss(output, y).mean()

        result = dict()
        for i, t in enumerate(self.targets):
            t_var = torch.full_like(y, t)
            x_p = trainer.eval_attack.perturb(trainer.model, x, t_var)
            with torch.no_grad():
                result[f"out_p{i}"] = trainer.model(x_p)
                result[f"loss_p_targ{i}"] = trainer.loss(result[f"out_p{i}"],
                                                         t_var).mean().item()
                result[f"loss_p{i}"] = trainer.loss(result[f"out_p{i}"], y).mean().item()

        return NameDict(x=x, target=y, out=output, loss=loss.item(), x_p=x_p, **result)


# Autoencoder

def autoencoder_train_step(trainer, batch):
    trainer.model.train()
    x = batch[0]
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
        """Copied from ignite/examples/gan/dcgan and modified"""
        trainer.model.train()
        real = batch[0]
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
