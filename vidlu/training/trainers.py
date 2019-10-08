import collections
from argparse import Namespace
from collections import Callable, Mapping
from functools import partial, lru_cache
import dataclasses as dc
from dataclasses import dataclass, InitVar

from tqdm import tqdm
import torch
from torch import nn

from vidlu import modules as M
from vidlu.data import Record
from vidlu.data_utils import DataLoader
from vidlu.training.lr_schedulers import ConstLR
from vidlu.utils.func import default_args, params, Empty, ArgTree, argtree_partial
from vidlu.utils.collections import NameDict
from vidlu.training.engine import Engine


def default_prepare_batch(batch, feature_type=torch.Tensor, device=None, non_blocking=False):
    """ A function for putting feature batches on the relevant device"""

    def _prepare(x):
        return x.to(device=device, non_blocking=non_blocking)

    if isinstance(batch, feature_type):
        return _prepare(batch)
    elif isinstance(batch, (collections.Mapping, Record)):
        return type(batch)({k: _prepare(x) for k, x in batch.items()})
    elif isinstance(batch, collections.Sequence):
        return type(batch)(_prepare(x) for x in batch)


# Evaluator and trainer ############################################################################


def extend_output(output, *x):
    return output, NameDict(prediction=output)


class Missing:
    def __init__(self):
        raise TypeError('`Missing` constructor was called, indicating that a parameter with default'
                        + ' value `Missing` has not been assigned a "real" value.')


class _MetricsMixin:
    def add_metric(self, m):
        self.metrics[type(m).__name__] = m

    def _reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def _update_metrics(self, es):
        for m in self.metrics.values():
            m.update(es.output)

    def get_metric_values(self, *, reset=False):
        metric_evals = dict()
        for k, m in self.metrics.items():
            value = m.compute()
            metric_evals.update(value if isinstance(value, dict) else {k: value.compute()})
        if reset:
            self._reset_metrics()
        return metric_evals


@dataclass
class Evaluator(_MetricsMixin):
    model: Callable = Missing
    loss_f: InitVar[Callable] = Missing
    prepare_batch: Callable = default_prepare_batch
    data_loader_f: Callable = partial(DataLoader, num_workers=4)
    batch_size: int = 1
    metrics: dict = dc.field(default_factory=dict)
    extend_output: Callable = extend_output
    eval_step: Callable = Missing

    loss: Callable = dc.field(init=False)

    def __post_init__(self, loss_f):
        self.prepare_batch = partial(self.prepare_batch, device=M.utils.get_device(self.model),
                                     non_blocking=False)
        self.eval_step = partial(self.eval_step, self)
        self.loss = loss_f()
        self.metrics = dict()
        if not isinstance(self.metrics, Mapping):
            for m in self.metrics:
                self.add_metric(m)

        self.evaluation = Engine(lambda e, b: self.eval_step(b))
        self.evaluation.started.add_handler(lambda _: self._reset_metrics())
        self.evaluation.iteration_completed.add_handler(self._update_metrics)

    def eval(self, dataset, batch_size=None):
        return self.evaluation.run(tqdm(
            self.data_loader_f(dataset, batch_size=batch_size or self.batch_size, shuffle=False,
                               drop_last=False)))


# Trainer ##########################################################################################


@dataclass
class Trainer(Evaluator):
    state_attrs = ('model', 'training', 'optimizer', 'lr_scheduler')

    eval_batch_size: int = None
    weight_decay: float = None  # R
    optimizer_f: InitVar[Callable] = None  # O
    epoch_count: int = Missing  # O
    lr_scheduler_f: InitVar[Callable] = ConstLR  # O
    train_step: Callable = Missing  # O
    jitter: Callable = None  # D
    optimizer_maker: InitVar[Callable] = None
    extension: object = dc.field(default_factory=Namespace)  # D

    optimizer: object = dc.field(init=False)
    lr_scheduler: object = dc.field(init=False)

    def __post_init__(self, loss_f, optimizer_f, lr_scheduler_f, optimizer_maker):
        super().__post_init__(loss_f)

        if optimizer_maker is None:
            if self.weight_decay is None:
                self.weight_decay = params(optimizer_f).weight_decay
            elif params(optimizer_f).weight_decay not in [0, Empty]:
                raise ValueError("The parameter weight_decay of optimizer_f should be unassigned.")
            self.optimizer = optimizer_f(self.model.parameters(), weight_decay=self.weight_decay)
        else:
            self.optimizer = optimizer_maker(self, optimizer_f)

        if 'epoch_count' in params(lr_scheduler_f):
            if params(lr_scheduler_f).epoch_count is not Empty:
                raise ValueError(
                    "The parameter epoch_count of lr_scheduler_f should be unassigned.")
            lr_scheduler_f = partial(lr_scheduler_f, epoch_count=self.epoch_count)
        self.lr_scheduler = lr_scheduler_f(optimizer=self.optimizer)

        self.train_step = partial(self.train_step, self)

        self.training = Engine(lambda e, b: self.train_step(b))
        self.training.epoch_completed.add_handler(lambda e: self.lr_scheduler.step())
        self.training.epoch_started.add_handler(lambda e: self._reset_metrics())
        self.training.iteration_completed.add_handler(self._update_metrics)

    def train(self, dataset, restart=False):
        data_loader = self.data_loader_f(dataset.map(self.jitter) if self.jitter else dataset,
                                         batch_size=self.batch_size, shuffle=True, drop_last=True)
        return self.training.run(data_loader, max_epochs=self.epoch_count, restart=restart)

    def eval(self, dataset, batch_size=None):
        super().eval(dataset, batch_size=self.eval_batch_size)

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in type(self).state_attrs}

    def load_state_dict(self, state_dict):
        for k in type(self).state_attrs:
            getattr(self, k).load_state_dict(state_dict[k])


@dataclass
class AdversarialTrainer(Trainer):
    attack_f: InitVar[callable] = Missing
    eval_attack_f: InitVar[callable] = None

    attack: object = dc.field(init=False)
    eval_attack: object = dc.field(init=False)

    def __post_init__(self, loss_f, optimizer_f, lr_scheduler_f, optimizer_maker, attack_f,
                      eval_attack_f):
        super().__post_init__(loss_f, optimizer_f, lr_scheduler_f, optimizer_maker)
        if attack_f is Missing:
            raise ValueError(f"`{type(self).__name__}` missing argument `attack_f`.")
        self.attack = attack_f(model=self.model, **(
            dict(loss=self.loss) if params(attack_f)['loss'] is Empty else {}))
        if isinstance(eval_attack_f, ArgTree):
            eval_attack_f = argtree_partial(attack_f, eval_attack_f)
        eval_attack_f = eval_attack_f or attack_f
        self.eval_attack = eval_attack_f(model=self.model, **(
            dict(loss=self.loss) if params(eval_attack_f)['loss'] is Empty else {}))


# GAN ##############################################################################################

class GANTrainer(Trainer):
    def __init__(self, model, loss=None, optimizer_f=None, epoch_count=1,
                 prepare_batch=None, device=None, non_blocking=False):
        loss = loss or nn.BCELoss()
        args = {k: v for k, v in locals() if k not in ['self', 'class']}
        super().__init__(**args)

    @lru_cache(1)
    def _get_real_labels(self, batch_size):
        return torch.ones(batch_size, device=self.model.device)

    @lru_cache(1)
    def _get_fake_labels(self, batch_size):
        return torch.zeros(batch_size, device=self.model.device)

    def train_step(self, batch):
        """ Copied from ignite/examples/gan/dcgan and modified"""
        self.model.train()
        real = self.prepare_batch(batch)[0]
        batch_size = real.shape[0]
        real_labels = self._get_real_labels(batch_size)

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) ##########
        discriminator, generator = self.model.discriminator, self.model.generator
        discriminator.zero_grad()

        # training discriminator with real
        output = discriminator(real)
        errD_real = self.loss(output, real_labels)
        D_real = output.mean().item()
        errD_real.backward()

        fake = generator(self.model.sample_z(batch_size))

        # training discriminator with fake
        output = discriminator(fake.detach())
        errD_fake = self.loss(output, self._get_fake_labels(batch_size))
        D_fake1 = output.mean().item()
        errD_fake.backward()

        self.optimizer['D'].step()

        # (2) Update G network: maximize log(D(G(z))) ##########################
        generator.zero_grad()

        # Update generator. We want to make a step that will make it more likely that D outputs "real"
        output = discriminator(fake)
        errG = self.loss(output, real_labels)
        D_fake2 = output.mean().item()

        errG.backward()
        self.optimizer['G'].step()

        return NameDict(errD=(errD_real + errD_fake).item(), errG=errG.item(), D_real=D_real,
                        D_fake1=D_fake1, D_fake2=D_fake2)
