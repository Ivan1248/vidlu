import copy
import typing as T
from vidlu.utils.func import partial
import dataclasses as dc
from dataclasses import dataclass, InitVar
import logging
from warnings import warn
import os
import random

from tqdm import tqdm
import torch
import numpy as np

import vidlu.utils.distributed as vud
from vidlu.data import DataLoader, BatchTuple
import vidlu.data.utils as vdu
import vidlu.modules.utils as vmu
from vidlu.optim.lr_schedulers import ConstLR
from vidlu.utils.func import params, Empty, Required
from vidlu.utils.collections import NameDict
from vidlu.utils.misc import Event, Stopwatch, broadcast
import vidlu.configs.training as vct
from vidlu.training.extensions import TrainerExtension


# EpochLoop based on Ignite Engine #################################################################

def _to_hours_mins_secs(t_s):
    """Convert seconds to hours, minutes, and seconds."""
    m, s = divmod(t_s, 60)
    h, m = divmod(m, 60)
    return h, m, s


class State(NameDict):
    """An object that is used to pass internal and user-defined state between event handlers"""

    def __init__(self, **kwargs):
        super().__init__()
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self.iteration = -1
        self.epoch = -1
        self.result = None
        self.batch = None
        self.batch_count = None
        self.update(**kwargs)


class EpochLoop(object):
    """Runs a given process_function over each batch and emits events..

    Based on Ignite Engine (https://pytorch.org/ignite).

    Args:
        iter_procedure (Callable): A procedure receiving a handle to the engine
            and the current batch in each iteration, and returns data to be
            stored in the engine's state.

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = EpochLoop(train_and_store_loss)
        engine.run(data_loader)
    """

    def __init__(self, iter_procedure):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.logger.addHandler(logging.NullHandler())
        self.iter_procedure = iter_procedure
        self.should_terminate = False
        self.should_terminate_epoch = False
        self.state = State()

        # events
        self.started = Event()
        self.completed = Event()
        self.epoch_started = Event()
        self.epoch_completed = Event()
        self.iter_started = Event()
        self.iter_completed = Event()

    def terminate(self):
        """Sends a signal for completely terminating after the current iteration.
        """
        self.logger.info(
            "Terminate signaled. Epoch loop will stop after current iteration is finished.")
        self.should_terminate = True

    def terminate_epoch(self):
        """Sends a signal for terminating the current epoch after the current iteration.
        """
        self.logger.info("Terminate current epoch is signaled. Current epoch iteration will stop"
                         + " after current iteration is finished.")
        self.should_terminate_epoch = True

    def _run_once_on_dataset(self):
        for self.state.iteration, batch in enumerate(self.state.data_loader):
            try:
                self.state.batch = batch
                self.iter_started(self.state)
                self.state.result = self.iter_procedure(self, batch)
                self.iter_completed(self.state)
                del self.state.batch, self.state.result
            finally:
                if self.should_terminate or self.should_terminate_epoch:
                    self.should_terminate = self.should_terminate_epoch = False
                    return True

    def run(self, data, max_epochs=1, restart=True, **kwargs):
        """Runs the `process_function` over the passed data.

        Args:
            data (Iterable): 1 or more collections of batches allowing repeated
                iteration (e.g. list or `DataLoader`).
            max_epochs (int, optional): max epochs to run for (default: 1)
            restart (bool, optional): whether to reset the training state before
                running. Default: `True`.
            **kwargs (dict): additional data to be stored in the state.
        Returns:
            State: output state
        """
        if restart or self.state is None:
            self.state.reset(metrics={})

        self.state.update(data_loader=data, max_epochs=max_epochs, batch_count=len(data), **kwargs)

        self.logger.info(f"Epoch loop run starting with max_epochs={max_epochs}.")
        with Stopwatch() as sw_total:
            self.started(self.state)

            if self.state.epoch + 1 >= max_epochs:
                warn("All epochs are already completed.")

            while self.state.epoch + 1 < max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self.epoch_started(self.state)

                with Stopwatch() as sw_epoch:
                    self._run_once_on_dataset()

                hours, mins, secs = _to_hours_mins_secs(sw_epoch.time)
                self.logger.info(
                    f"Epoch {self.state.epoch} completed after {hours:02}:{mins:02}:{secs:02}.")
                self.epoch_completed(self.state)

            self.completed(self.state)

        hours, mins, secs = _to_hours_mins_secs(sw_total.time)
        self.logger.info(f"Epoch loop run completed after {hours:02}:{mins:02}:{secs:02}.")

        return self.state

    def state_dict(self):
        return dict(epoch=self.state.epoch)

    def load_state_dict(self, state_dict):
        self.state.epoch = state_dict['epoch']


# Batch preparation ################################################################################

def default_prepare_batch(batch, feature_type=torch.Tensor, device=None, non_blocking=False):
    """A function for putting feature batches on the relevant device"""

    def _prepare(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, non_blocking=non_blocking)
        return x

    if isinstance(batch, feature_type):
        return _prepare(batch)
    elif hasattr(type(batch), "items"):
        return type(batch)({k: _prepare(x) for k, x in batch.items()})
    elif isinstance(batch, BatchTuple):
        return BatchTuple(default_prepare_batch(b, feature_type, device, non_blocking)
                          for b in batch)
    elif isinstance(batch, T.Sequence):
        return type(batch)(_prepare(x) for x in batch)
    raise TypeError(f"Invalid batch type {type(batch)}")


# Evaluator and trainer ############################################################################

NUM_WORKERS = int(os.environ.get("VIDLU_NUM_WORKERS",
                                 max(4, min(12, os.cpu_count() // 8))))  # TODO: per GPU
print(f"{NUM_WORKERS=}")


def deterministic_data_loader_args(distributed):
    if distributed:
        raise NotImplementedError(
            "Deterministic DataLoader arguments not available for distributed training")

    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2 ** 32
        np.random.seed(seed)
        random.seed(seed)

    g = torch.Generator()
    g.manual_seed(0)
    return dict(worker_init_fn=worker_init_fn, generator=g)


@dataclass
class Evaluator:
    model: T.Callable = Required
    loss: T.Callable = Required
    prepare_batch: T.Callable = default_prepare_batch
    data_loader_f: vdu.TMultiDataLoaderF = partial(
        vdu.auto_data_loader, dl_f=DataLoader, multi_dl_f='zip', num_workers=NUM_WORKERS,
        shuffle=True)
    deterministic: bool = False
    batch_size: T.Union[int, T.Sequence[int]] = 1
    metrics: list = dc.field(default_factory=list)
    eval_step: T.Callable = Required
    eval_sync: bool = bool(
        int(os.environ.get("VIDLU_SYNCHRONIZE_STEP", 1))) and torch.cuda.is_available()
    distributed: bool = False

    def __post_init__(self):
        self.prepare_batch = partial(self.prepare_batch, device=vmu.get_device(self.model),
                                     non_blocking=False)

        def put_metrics_into_state():
            self.evaluation.state.metrics = self.get_metric_values()

        def evaluation(engine, batch):
            return self._run_step(self.eval_step, batch, synchronize=self.eval_sync)

        self.evaluation = EpochLoop(evaluation)
        self.evaluation.started.add_handler(lambda _: self._reset_metrics())
        self.evaluation.epoch_completed.add_handler(lambda _: put_metrics_into_state())
        self.evaluation.iter_completed.add_handler(self._update_metrics)

    @torch.no_grad()
    def _reset_metrics(self):
        for m in self.metrics:
            m.reset()

    @torch.no_grad()
    def _update_metrics(self, state):
        for m in self.metrics:
            m.update(state.result)

    @torch.no_grad()
    def get_metric_values(self, *, reset=False):
        metric_evals = dict()
        for m in self.metrics:
            value = m.compute()
            metric_evals.update(value if isinstance(value, dict) else {m.name: value.compute()})
        if reset:
            self._reset_metrics()
        return metric_evals

    def _run_step(self, step, batch, synchronize=False):
        batch = self.prepare_batch(batch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if synchronize:
            torch.cuda.synchronize()
        with Stopwatch() as sw:
            output = step(self, batch)
            if synchronize:
                torch.cuda.synchronize()
        output['freq'] = 1 / sw.time
        if isinstance(output, T.MutableMapping) and torch.cuda.is_available():
            output['mem'] = torch.cuda.max_memory_allocated() // 2 ** 20
        return output

    def _make_data_loader(self, *datasets, batch_size, **kwargs):
        """Creates a dataloader based on a data loader factory, batch size, and other potential
        arguments.

        If distributed training is required, the samplers in the data loader are modified and batch
        size is divided by the number of processes.
        """
        if self.distributed:
            batch_size = divide_batch_size_over_processes(batch_size)
        data_loader = self.data_loader_f(*datasets, batch_size=batch_size, **kwargs)
        if self.distributed:
            data_loader = vdu.make_data_loader_distributed(data_loader)
        return data_loader

    def eval(self, *datasets, batch_size=None, **kwargs):
        dl_kwargs = dict(drop_last=False, batch_size=batch_size or self.batch_size, shuffle=False)
        if self.deterministic:
            dl_kwargs.update(deterministic_data_loader_args(self.distributed))
        data_loader = self._make_data_loader(*datasets, **dl_kwargs)

        return self.evaluation.run(tqdm(data_loader), **kwargs)


def divide_batch_size_over_processes(batch_size: T.Union[int, T.Sequence[int]]):
    if isinstance(batch_size, int):
        num_processes = vud.get_global_size()
        if batch_size % num_processes != 0:
            raise ValueError(
                f"{batch_size=} is not a multiple of the number of processes ({num_processes}).")
        return batch_size // num_processes
    return tuple(map(divide_batch_size_over_processes, batch_size))


# Trainer ##########################################################################################

# TODO: support for distributed training for metrics and iteration step proedures
@dataclass
class Trainer(Evaluator):
    """A class encapsulating all machine learning algorithm components.

    Additional state should be stored in in the `trainer.training.state`
    dictionary or in a training extension from `trainer.extensions`.
    """
    state_dict_attrs = ('model', 'training', 'optimizer', 'lr_scheduler', 'train_step', 'eval_step')

    eval_batch_size: T.Union[int, T.Sequence[int]] = None

    epoch_count: int = Required  # optimization
    optimizer_f: T.Callable = None  # optimization; vidlu.optim
    lr_scheduler_f: T.Callable = ConstLR  # optimization; vidlu.optim.lr_schedulers
    jitter: T.Callable = None  # learning
    train_step: T.Optional[T.Callable] = Required  # learning; vidlu.training.steps
    extension_fs: InitVar[T.Sequence[T.Callable]] = ()  # learning

    optimizer: T.Any = dc.field(init=False)
    lr_scheduler: T.Any = dc.field(init=False)
    extensions: T.Sequence[TrainerExtension] = dc.field(init=False)

    def __post_init__(self, extension_fs):
        super().__post_init__()

        if self.eval_step is None:
            self.eval_step = self._get_eval_step()

        self.optimizer = self.optimizer_f(
            self.model if isinstance(self.optimizer_f,
                                     vct.OptimizerMaker) else self.model.parameters())

        lr_scheduler_f = self.lr_scheduler_f
        if 'epoch_count' in params(lr_scheduler_f):
            if params(lr_scheduler_f).epoch_count is not Empty:
                raise ValueError(
                    "The parameter epoch_count of lr_scheduler_f should be unassigned.")
            lr_scheduler_f = partial(lr_scheduler_f, epoch_count=self.epoch_count)
        self.lr_scheduler = lr_scheduler_f(optimizer=self.optimizer)

        self.training = EpochLoop(lambda e, b: self._run_step(self.train_step, b))
        self.training.epoch_completed.add_handler(lambda e: self.lr_scheduler.step())
        self.training.epoch_started.add_handler(lambda e: self._reset_metrics())
        self.training.iter_completed.add_handler(self._update_metrics)

        self.extensions = [e() for e in extension_fs]
        if len(set(map(type, self.extensions))) < len(self.extensions):
            raise RuntimeError("Multiple extensions of the same type are not allowed. The types are"
                               + f"{', '.join([type(e).__name__ for e in self.extensions])}.")
        for e in self.extensions:
            e.initialize(self)

        self._initialized = True

    def _get_eval_step(self):
        result = self.__dict__['eval_step']
        if result is None:
            if not hasattr(self.train_step, 'eval'):
                result = partial(self.train_step.__call__, eval=True)
            else:
                result = copy.copy(self.train_step)
                result.eval = True
        return result

    def train(self, *datasets, restart=False):
        if self.jitter is not None:
            jitters = broadcast(self.jitter, len(datasets))
            datasets_jitt = [ds.map(jitter) for jitter, ds in zip(jitters, datasets)]
        else:
            datasets_jitt = datasets

        dl_kwargs = dict(drop_last=True, batch_size=self.batch_size)
        if self.deterministic:
            dl_kwargs.update(deterministic_data_loader_args(self.distributed))
        data_loader = self._make_data_loader(*datasets_jitt, **dl_kwargs)

        return self.training.run(data_loader, max_epochs=self.epoch_count, restart=restart)

    def eval(self, *datasets, batch_size=None, **kwargs):
        return super().eval(*datasets,
                            batch_size=self.eval_batch_size if batch_size is None else batch_size,
                            **kwargs)

    def state_dict(self):
        return dict(**{k: attr.state_dict() for k in type(self).state_dict_attrs
                       if (hasattr(attr := getattr(self, k), "state_dict"))},
                    extensions={f"{type(e)}": e.state_dict() for e in self.extensions})

    def load_state_dict(self, state_dict):
        for k in self.state_dict_attrs:
            if hasattr(attr := getattr(self, k), "state_dict"):
                attr.load_state_dict(state_dict[k])
        for e in self.extensions:
            e.load_state_dict(state_dict['extensions'][f"{type(e)}"])

    def __getattr__(self, key):
        for e in self.extensions:
            if hasattr(e, key):
                return getattr(e, key)
        raise AttributeError(f'Neither the `Trainer` object nor its extensions'
                             + f' have a "{key}" attribute.')

    def __setattr__(self, key, value):
        if '_initialized' not in self.__dict__ or key in self.__dict__:
            self.__dict__[key] = value
            return
        for e in self.extensions:
            if hasattr(e, key):
                return setattr(e, key, value)
        raise AttributeError(f'Neither the `Trainer` object nor its extensions'
                             + f' have a "{key}" attribute.')
