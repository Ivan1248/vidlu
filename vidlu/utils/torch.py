import contextlib

import torch


@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


@contextlib.contextmanager
def save_grads(params):
    param_grads = [(p, p.grad.clone().detach()) for p in params]
    yield
    for p, g in param_grads:
        p.grad = g


def profile(func, on_cuda=True):
    on_cuda = True
    with torch.autograd.profiler.profile(use_cuda=on_cuda) as prof:
        output = func()
    if on_cuda:
        torch.cuda.synchronize()
    return output, prof.key_averages().table('cuda_time_total')


# Math

def to_one_hot(labels, c):
    one_hot = torch.cuda.FloatTensor(labels.size(0), c).zero_()
    target = one_hot.scatter_(1, labels.unsqueeze(-1), 1)
    return target

