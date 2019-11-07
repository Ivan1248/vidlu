from typing import Union, Callable, Mapping, Sequence
from functools import partial
import dataclasses as dc
from dataclasses import dataclass, InitVar

from tqdm import tqdm
import torch

from vidlu import modules as M
from vidlu.data import Record, DataLoader, ZipDataLoader, BatchTuple
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
    elif isinstance(batch, (Mapping, Record)):
        return type(batch)({k: _prepare(x) for k, x in batch.items()})
    elif isinstance(batch, BatchTuple):
        return BatchTuple(default_prepare_batch(b, feature_type, device, non_blocking)
                          for b in batch)
    elif isinstance(batch, Sequence):
        return type(batch)(_prepare(x) for x in batch)
    raise TypeError(f"Invalid batch type {type(batch)}")


# Evaluator and trainer ############################################################################


def extend_output(output):
    return output, NameDict(prediction=output)


class Missing:
    def __init__(self):
        raise TypeError('`Missing` constructor was called, indicating that a parameter with default'
                        + ' value `Missing` has not been assigned a "real" value.')


class _MetricsMixin:
    def add_metric(self, m):
        self.metrics.append(m)

    def _reset_metrics(self):
        for m in self.metrics:
            m.reset()

    def _update_metrics(self, state):
        for m in self.metrics:
            m.update(state.output)

    def get_metric_values(self, *, reset=False):
        metric_evals = dict()
        for m in self.metrics:
            value = m.compute()
            metric_evals.update(value if isinstance(value, dict) else {m.name: value.compute()})
        if reset:
            self._reset_metrics()
        return metric_evals


@dataclass
class Evaluator(_MetricsMixin):
    model: Callable = Missing
    loss_f: InitVar[Callable] = Missing
    prepare_batch: Callable = default_prepare_batch
    data_loader_f: Callable = partial(DataLoader, num_workers=3)
    batch_size: int = 1
    metrics: dict = dc.field(default_factory=list)
    extend_output: Callable = extend_output
    eval_step: Callable = Missing

    loss: Callable = dc.field(init=False)

    def __post_init__(self, loss_f):
        self.prepare_batch = partial(self.prepare_batch, device=M.utils.get_device(self.model),
                                     non_blocking=False)
        self.eval_step = partial(self.eval_step, self)
        self.loss = loss_f()

        self.evaluation = Engine(lambda e, b: self.eval_step(b))
        self.evaluation.started.add_handler(lambda _: self._reset_metrics())
        self.evaluation.iteration_completed.add_handler(self._update_metrics)

    @staticmethod
    def _broadcast(obj, n):
        if isinstance(obj, Sequence):
            if len(obj) != n:
                raise RuntimeError(f"`obj` already is a `Sequence` but its size"
                                   + f" ({len(obj)}) is not `n` = {n}.")
            return obj
        return [obj] * n

    def get_data_loader(self, *datasets, batch_size, shuffle, drop_last, **kwargs):
        data_loader_fs = self._broadcast(self.data_loader_f, len(datasets))
        batch_sizes = self._broadcast(batch_size, len(datasets))
        data_loaders = [dl_f(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last, **kwargs)
                        for ds, dl_f, bs in zip(datasets, data_loader_fs, batch_sizes)]
        return data_loaders[0] if len(datasets) == 1 else ZipDataLoader(*data_loaders)

    def eval(self, *datasets, batch_size=None):
        data_loader = self.get_data_loader(*datasets, batch_size=batch_size or self.batch_size,
                                           shuffle=True, drop_last=False)
        return self.evaluation.run(tqdm(data_loader))


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
    extension_fs: InitVar[Sequence] = ()  # D

    optimizer: object = dc.field(init=False)
    lr_scheduler: object = dc.field(init=False)
    extensions: Sequence = dc.field(init=False)

    def __post_init__(self, loss_f, optimizer_f, lr_scheduler_f, optimizer_maker, extension_fs):
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

        self.extensions = [e() for e in extension_fs]
        if len(set(map(type, self.extensions))) < len(self.extensions):
            raise RuntimeError("Multiple extensions of the same type are not allowed. The types are"
                               + f"{', '.join([type(e).__name__ for e in self.extensions])}.")
        for e in self.extensions:
            e.initialize(self)

        self._initialized = True

    def train(self, *datasets, restart=False):
        data_loader = self.get_data_loader(*datasets, batch_size=self.batch_size,
                                           shuffle=True, drop_last=True)
        return self.training.run(data_loader, max_epochs=self.epoch_count, restart=restart)

    def eval(self, *datasets, batch_size=None):
        super().eval(*datasets,
                     batch_size=self.eval_batch_size if batch_size is None else batch_size)

    def state_dict(self):
        return {k: getattr(self, k).state_dict() for k in type(self).state_attrs}

    def load_state_dict(self, state_dict):
        for k in type(self).state_attrs:
            getattr(self, k).load_state_dict(state_dict[k])

    def __getattr__(self, key):
        for e in self.extensions:
            if hasattr(e, key):
                return getattr(e, key)
        raise AttributeError(f'Neither the `Trainer` object nor its extensions'
                             + f' have a "{key}" attribute.')

    def __setattr__(self, key, value):
        if '_initialized' not in self.__dict__:
            self.__dict__[key] = value
            return
        for e in self.extensions.values():
            if hasattr(e, key):
                return setattr(e, key, value)
        raise AttributeError(f'Neither the `Trainer` object nor its extensions'
                             + f' have a "{key}" attribute.')
