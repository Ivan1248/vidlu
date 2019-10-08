import contextlib

import torch


@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch)
    yield
    model.apply(switch)


@contextlib.contextmanager
def save_grads(params):
    param_grads = [(p, p.grad.detach().clone()) for p in params]
    yield
    for p, g in param_grads:
        p.grad = g


def fuse_tree_batches(*args, tree_type=None):
    tree_type = tree_type or type(args[0])
    keys = args[0].keys()
    values = []
    for k in keys:
        vals = [a[k] for a in args]
        if isinstance(vals[0], tree_type):
            values.append(fuse_tree_batches(*vals, tree_type=tree_type))
        else:
            values.append(torch.cat(vals, dim=0))
    return tree_type(zip(keys, values))


def reset_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
