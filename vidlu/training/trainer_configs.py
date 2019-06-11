import dataclasses as dc
from collections import Callable
from dataclasses import dataclass, InitVar
from functools import partial

from ignite._utils import convert_tensor
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from vidlu import modules
from vidlu.data_utils import DataLoader
from vidlu.training.lr_schedulers import ScalableMultiStepLR, ScalableLambdaLR
from vidlu.utils.func import params, Empty


def default_prepare_batch(batch, device=None, non_blocking=False):
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


@dataclass
class EvaluatorConfig:
    model: Callable = Empty
    loss_f: InitVar = Empty
    prepare_batch = default_prepare_batch
    data_loader_f = partial(DataLoader, batch_size=1, num_workers=2)

    loss: type(default_prepare_batch) = dc.field(init=False)

    def check_all_initialized(self):
        for k, v in self.__dict__:
            if v is Empty:
                raise TypeError(f"Field {k} is missing.")

    def __post_init__(self, loss_f):
        self.loss = loss_f()
        self.prepare_batch = partial(self.prepare_batch,
                                     device=modules.utils.get_device(self.model),
                                     non_blocking=False)
        self.check_all_initialized()


@dataclass
class TrainerConfig(EvaluatorConfig):
    weight_decay: float = Empty
    optimizer_f: InitVar = Empty
    epoch_count: int = Empty
    lr_scheduler_f: InitVar = partial(LambdaLR, lr_lambda=lambda e: 1)
    batch_size: int = 1

    optimizer: object = dc.field(init=False)
    lr_scheduler: object = dc.field(init=False)

    def __post_init__(self, loss_f, optimizer_f, lr_scheduler_f):
        super().__post_init__(loss_f)

        if params(optimizer_f).get('weight_decay', Empty) is not Empty:
            raise ValueError(
                "The `weight_decay` parameter of `optimizer_f` should be unassigned"
                + " because it is a parameter of `TrainerConfig`.")
        if 'epoch_count' in params(lr_scheduler_f):
            if params(lr_scheduler_f).epoch_count is not Empty:
                raise ValueError(
                    "The `epoch_count` parameter of `lr_scheduler_f` should be unassigned"
                    + " because it is a parameter of `TrainerConfig`.")
            lr_scheduler_f = partial(lr_scheduler_f, epoch_count=self.epoch_count)
        self.optimizer = optimizer_f(params=self.model.parameters(), weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler_f(optimizer=self.optimizer)


@dataclass
class ResNetCifarTrainerConfig(TrainerConfig):
    # as in www.arxiv.org/abs/1603.05027
    optimizer_f: InitVar = partial(SGD, lr=1e-1, momentum=0.9, weight_decay=Empty)
    weight_decay: float = 1e-4  # annotat
    epoch_count: int = 200
    lr_scheduler_f: InitVar = partial(ScalableMultiStepLR, milestones=[0.3, 0.6, 0.8], gamma=0.2)
    batch_size: int = 128


@dataclass
class WRNCifarTrainerConfig(ResNetCifarTrainerConfig):
    # as in www.arxiv.org/abs/1603.05027
    weight_decay: float = 5e-4


@dataclass
class DenseNetCifarTrainerConfig(TrainerConfig):
    # as in www.arxiv.org/abs/1608.06993
    weight_decay: float = 1e-4
    optimizer_f: InitVar = partial(SGD, lr=1e-1, momentum=0.9, weight_decay=Empty, nesterov=True)
    epoch_count: int = 100
    lr_scheduler_f: InitVar = partial(ScalableMultiStepLR, milestones=[0.5, 0.75], gamma=0.1)
    batch_size: int = 64


@dataclass
class SmallImageClassifierTrainerConfig(TrainerConfig):
    # as in www.arxiv.org/abs/1603.05027
    weight_decay: float = 0
    optimizer_f: InitVar = partial(SGD, lr=1e-2, momentum=0.9)
    epoch_count: int = 50
    batch_size: int = 64


@dataclass
class LadderDenseNetTrainerConfig(TrainerConfig):
    # custom
    weight_decay: float = 1e-4
    optimizer_f: InitVar = partial(SGD, lr=5e-4, momentum=0.9, weight_decay=Empty)
    lr_scheduler_f: InitVar = partial(ScalableLambdaLR, lr_lambda=lambda p: (1 - p) ** 1.5)
    epoch_count: int = 40
    batch_size: int = 4


@dataclass
class GANTrainerConfig(TrainerConfig):
    loss_f: Callable = nn.BCELoss
