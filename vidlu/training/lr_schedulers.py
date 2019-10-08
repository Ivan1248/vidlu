from functools import partial

from torch.optim import lr_scheduler

from vidlu.utils.func import default_args

'''
class MultiStepLR(lr_scheduler.MultiStepLR):
    """Improved MultiStepLR where get_lr always returns the correct value.
    Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.new_lrs = None
        self.last_milestone_update = last_epoch  # for valid lr_value when get_lr is called from outside
        super().__init__(optimizer=optimizer, milestones=milestones, gamma=gamma,
                         last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            if self.last_milestone_update != self.last_epoch:
                self.last_milestone_update = self.last_epoch
                self.new_lrs = [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                                for group in self.optimizer.param_groups]
            return self.new_lrs
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
'''


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


class ConstLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def get_lr(self):
        return self.base_lrs


class ScalableLambdaLR(lr_scheduler.LambdaLR):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor from the epoch index divided by the total number of epochs (a
            number between 0 and 1), or a list of such functions, one for each
            group in optimizer.param_groups.
        epoch_count: The total number of epochs.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> epoch_count = 100
        >>> lambda1 = lambda progress: 1/30 * progress
        >>> lambda2 = lambda progress: 0.95 ** progress
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2], epoch_count=epoch_count)
        >>> for epoch in range(epoch_count):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, lr_lambda, epoch_count,
                 last_epoch=default_args(lr_scheduler.LambdaLR).last_epoch):
        if not isinstance(lr_lambda, (list, tuple)):
            lr_lambda = [lr_lambda] * len(optimizer.param_groups)
        lr_lambda = [lambda e: ll(e / epoch_count) for ll in lr_lambda]
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


class CosineLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, epoch_count, eta_min=0,
                 last_epoch=default_args(lr_scheduler.LambdaLR).last_epoch):
        super().__init__(optimizer=optimizer, T_max=epoch_count, eta_min=eta_min,
                         last_epoch=last_epoch)
