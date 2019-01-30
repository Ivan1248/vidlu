from functools import partial, partialmethod, wraps

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.optim import SGD

from vidlu.utils.func import default_args, hard_partial
from vidlu.utils.misc import locals_from_first_initializer
from utils.func import Empty
from vidlu.utils.collections import NameDict


class TrainingConfig:
    def __init__(self, loss_f, weight_decay, epoch_count, batch_size, optimizer_f,
                 lr_scheduler_f=partial(LambdaLR, lr_lambda=lambda e: 1)):
        self.args = NameDict(locals())
        self.optimizer, self.lr_scheduler = [None] * 2
        assert 'optimizer' not in default_args(lr_scheduler_f)

    def finish_init(self, parameters):
        args = self.args
        self.optimizer = args.optimizer_f(parameters=parameters,
                                          weight_decay=args.weight_decay)
        self.lr_scheduler = args.lr_scheduler_f(optimizer=self.optimizer)


class ResNetCifar(TrainingConfig):
    __init__ = partialmethod(TrainingConfig.__init__, weight_decay=1e-4, epoch_count=200,
                             batch_size=128,
                             lr_scheduler_f=partial(MultiStepLR, milestones=[60, 120, 160],
                                                    gamma=0.2),
                             optimizer_f=partial(SGD, momentum=0.9))


class WRNCifar(TrainingConfig):
    __init__ = partialmethod(ResNetCifar.__init__, weight_decay=5e-4)


class DenseNetCifar(TrainingConfig):
    __init__ = partialmethod(TrainingConfig.__init__, weight_decay=1e-4, epoch_count=100,
                             batch_size=64,
                             lr_scheduler_f=partial(MultiStepLR, milestones=[50, 75], gamma=0.1),
                             optimizer_f=default_args(ResNetCifar).optimizer_f)
