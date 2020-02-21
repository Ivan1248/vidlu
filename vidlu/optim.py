import typing as T

import torch.optim
from torch.optim.optimizer import required, Optimizer
import numpy as np

from vidlu import ops


class GradientSignDescent(torch.optim.SGD):
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            weight_decay, momentum, dampening, nesterov = (
                group[k] for k in ['weight_decay', 'momentum', 'dampening', 'nesterov'])

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p.sign())

        return loss


def _identity(x):
    return x


class ProcessedGradientDescent(torch.optim.SGD):
    def __init__(self, params, lr=required, process_grad=None, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid step size: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, process_grad=process_grad, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        Optimizer.__init__(self, params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            weight_decay, momentum, dampening, nesterov, process_grad = (
                group[k] for k in
                ['weight_decay', 'momentum', 'dampening', 'nesterov', 'process_grad'])
            process_grad = process_grad or _identity

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.add_(-group['lr'], process_grad(d_p))

        return loss

def get_grad_processing(name) -> T.Callable:
    if name == 'sign':
        return lambda g: g.sign()
    elif name.startswith('normalize'):
        if '_' in name:
            p = name.split('_', 1)[1]
            p = np.inf if p == 'inf' else eval(p)
        else:
            p = 2
        return lambda g: g / ops.batch.norm(g, p, keep_dims=True)
    elif 'raw':
        return lambda g: g
