import math

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import *

from vidlu.utils.func import default_args
from vidlu.utils.misc import broadcast


class LRScheduler(lr_scheduler._LRScheduler):
    pass


class ScalableMultiStepLR(lr_scheduler.MultiStepLR):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of numbers from the interval (0, 1). Must be
            increasing. Milestone epoch indices are calculated by multiplying
            provided milestones with the total number of epochs (`epoch_count`).
        epoch_count: The total number of epochs.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> epoch_count = 100
        >>> scheduler = ScalableMultiStepLR(optimizer, milestones=[0.3, 0.8],
        ...                               epoch_count=epoch_count, gamma=0.1)
        >>> for epoch in range(epoch_count):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)

    Note:
        `MultiStepLR.get_lr` gives the wrong value (doubly decayed) if the
        learning rates were in the current epoch.
    """

    def __init__(self, optimizer, milestones, epoch_count, gamma=0.1,
                 last_epoch=default_args(lr_scheduler.MultiStepLR).last_epoch):
        super().__init__(optimizer, milestones=[round(m * epoch_count) for m in milestones],
                         gamma=gamma, last_epoch=last_epoch)


class ConstLR(LRScheduler):
    def get_lr(self):
        return self.base_lrs


class CosineLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, epoch_count, eta_min=0,
                 last_epoch=default_args(lr_scheduler.LambdaLR).last_epoch):
        super().__init__(optimizer=optimizer, T_max=epoch_count, eta_min=eta_min,
                         last_epoch=last_epoch)


class ScalableLR(lr_scheduler.LambdaLR):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        func (function or list): A function which computes a multiplicative
            factor from the epoch index divided by the total number of epochs (a
            number between 0 and 1), or a list of such functions, one for each
            group in optimizer.param_groups.
        epoch_count: The total number of epochs.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> epoch_count = 100
        >>> f1 = lambda progress: 1/30 * progress  # 0 <= progress < 1
        >>> f2 = lambda progress: 0.95 ** progress
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[f1, f2], epoch_count=epoch_count)
        >>> for epoch in range(epoch_count):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, func, epoch_count, scaling=1., min=0.,
                 last_epoch=default_args(lr_scheduler.LambdaLR).last_epoch):
        func, scaling, min = [broadcast(x, len(optimizer.param_groups))
                              for x in (func, scaling, min)]
        func = [lambda e: ll(0 if e == 0 else e / epoch_count) * s + m
                for ll, s, m in zip(func, scaling, min)]
        super().__init__(optimizer=optimizer, lr_lambda=func, last_epoch=last_epoch)


def quarter_cos(x):
    if not 0 <= x <= 1:
        raise ValueError(f"{x=} should be between 0 and 1.")
    return math.cos(x * math.pi / 2)


class QuarterCosLR(ScalableLR):
    def __init__(self, optimizer, epoch_count, min=0.,
                 last_epoch=default_args(lr_scheduler.LambdaLR).last_epoch):
        super().__init__(optimizer, func=quarter_cos, epoch_count=epoch_count, min=min,
                         last_epoch=last_epoch)
