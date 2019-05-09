import contextlib


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