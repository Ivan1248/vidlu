import contextlib

from ignite._utils import convert_tensor


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


def prepare_batch(batch, device=None, non_blocking=False):
    batch = tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)
    for x in batch:
        x.requires_grad = False
    return batch
