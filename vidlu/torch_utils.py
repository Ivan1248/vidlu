import contextlib
import typing as T
from pathlib import Path
import copy
import difflib
import logging
from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn
import torch.utils.checkpoint as tuc

import vidlu.utils.context as vuc


def is_float_tensor(x):
    return 'float' in str(x.dtype)


def is_int_tensor(x):
    return 'int' in str(x.dtype)


def round_float_to_int(x, dtype=torch.int):
    return x.sign().mul_(0.5).add_(x).to(dtype)


# General context managers (move out of utils/torch?

def switch_use_of_deterministic_algorithms(value):
    return vuc.switch_var(
        torch.are_deterministic_algorithms_enabled, torch.use_deterministic_algorithms, value,
        omit_unnecessary_calls=True)


# Context managers for modules and tensors

def preserve_attribute_tensor(objects, attrib_name):
    return vuc.preserve_attribute(objects, attrib_name, lambda x: x.detach().clone())


def switch_requires_grad(module_or_params, value):
    params = module_or_params.parameters() if isinstance(module_or_params, torch.nn.Module) \
        else module_or_params
    return vuc.switch_attribute(params, 'requires_grad', value)


def switch_training(module, value):
    """Sets the training attribute to false for the module and all submodules."""
    return vuc.switch_attribute(module.modules(), 'training', value)


@contextlib.contextmanager
def norm_stats_tracking_off(module):
    modules = [m for m in module.modules() if 'Norm' in type(m).__name__ and hasattr(m, 'momentum')]
    with vuc.switch_attribute(modules, 'momentum', 0), \
            preserve_attribute_tensor(modules, 'num_batches_tracked'):
        yield


@contextlib.contextmanager
def preserve_grads(params):
    param_grads = [(p, p.grad.detach().clone()) for p in params]
    yield param_grads
    for p, g in param_grads:
        p.grad = g


@contextlib.contextmanager
def preserve_params(params):
    param_param_saved = [(p, p.detach().clone()) for p in params]
    yield param_param_saved
    with torch.no_grad():
        for p, p_saved in param_param_saved:
            p.data = p_saved


@contextlib.contextmanager
def preserve_state(obj):
    state = copy.deepcopy(obj.state_dict())
    yield state
    obj.load_state_dict(state)


# tensor trees

def concatenate_tensors_trees(*args, tree_type=None):
    """Concatenates corresponding arrays from equivalent (parallel) trees.

    Example:
        >>> x1, y1, z1, x2, y2, z2 = [torch.tensor([i]) for i in range(6)]
        >>> cat = lambda x, y: torch.cat([x, y], dim=0)
        >>> a = dict(q=dict(x=x1, y=y1), z=z1)
        >>> b = dict(q=dict(x=x2, y=y2), z=z2)
        >>> c1 = concatenate_tensors_trees(a, b)
        >>> c2 = dict(q=dict(x=cat(x1, x2), y=cat(y1, y2)), z=cat(z1, z2))
        >>> assert c1 == c2

    Args:
        *args: trees with Torch arrays as leaves.
        tree_type (optional): A dictionary type used to recognize what a
            subtree is if they are of mixed types (e.g. `dict` and some other
            dictionary type that is not a subtype od `dict`).

    Returns:
        A tree with concatenated arrays
    """
    tree_type = tree_type or type(args[0])
    keys = args[0].keys()
    values = []
    for k in keys:
        vals = [a[k] for a in args]
        if isinstance(vals[0], tree_type):
            values.append(concatenate_tensors_trees(*vals, tree_type=tree_type))
        else:
            values.append(torch.cat(vals, dim=0))
    return tree_type(zip(keys, values))


# Autograd checkpointing

def is_modified(tensor: torch.Tensor):
    return tensor._version > 0


class StateAwareCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preservation_cm_f, preserve_rng_state, *args):
        result = tuc.CheckpointFunction.forward(ctx, run_function, preserve_rng_state, *args)
        if preservation_cm_f is not None:
            def preserving_run_function(*args):
                with preservation_cm_f():
                    return run_function(*args)

            ctx.run_function = preserving_run_function
        return result

    @staticmethod
    def backward(ctx, *args):
        return (None,) + tuc.CheckpointFunction.backward(ctx, *args)


def checkpoint_sa(function, *args, **kwargs):
    # Can preserve the state in calls to function in backward
    preservation_cm_f = kwargs.pop('preservation_cm_f', None)
    preserve_rng_state = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    return StateAwareCheckpointFunction.apply(function, preservation_cm_f, preserve_rng_state,
                                              *args)


def checkpoint_fix(function, *args, **kwargs):
    """A workaround for CheckpointFunctionBackward not being set (and called)
    when the output of `function` is an in-place modified view tensor, e.g
    `x[:].relu_()`.
    """

    def inplace_backward_fix_wrapper(*args_, **kwargs_):
        result = function(*args_, **kwargs_)
        if isinstance(result, torch.Tensor):
            return result[...] if is_modified(result) else result
        elif isinstance(result, T.Sequence):
            return type(result)(r[...] if is_modified(r) else r for r in result)
        elif isinstance(result, T.Mapping):
            return type(result)({k: r[...] if is_modified(r) else r for k, r in result.items()})
        else:
            raise NotImplementedError()

    return checkpoint_sa(inplace_backward_fix_wrapper, *args, **kwargs)


class StateAwareCheckpoint:  # preserves state in the first run
    def __init__(self, module, context_managers_fs=(norm_stats_tracking_off,)):
        self.context_managers_fs = context_managers_fs
        self.module = module if isinstance(module, nn.Module) else nn.ModuleList(module)

    def __call__(self, func, *args, **kwargs):
        with contextlib.ExitStack() as stack:
            for cmf in self.context_managers_fs:
                stack.enter_context(cmf(self.module))
            return checkpoint_fix(func, *args, **kwargs)


# Cuda memory management

def reset_cuda():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# Copyright (c) Facebook, Inc. and its affiliates. Code taken from detectron2.utils.memory
@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


# Copyright (c) Facebook, Inc. and its affiliates. Code taken from detectron2.utils.memory
def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


# AMP

class DummyGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    @staticmethod
    def update(new_scale=None):
        pass

    @staticmethod
    def get_scale():
        return 1.0

    @staticmethod
    def state_dict():
        return dict()

    @staticmethod
    def load_state_dict(*args, **kwargs):
        pass


# Profiling

def profile(func, on_cuda=True, device=None):
    with torch.autograd.profiler.profile(use_cuda=on_cuda) as prof:
        output = func()
    if on_cuda:
        torch.cuda.synchronize(device=device)
    return output, prof.key_averages().table('cuda_time_total')


# Algorithm comparison

def compare(a, b, name=None, key=None):
    import typing as T
    if isinstance(b, (str, Path)):
        name = b
    if name is not None:
        print(f"Comparing {name}")
        other = torch.load(b) if isinstance(b, (str, Path)) else b
        assert type(a) == type(other)
        result = compare(a, other)
        print("+" if result else "FAIL")
        return result
    if isinstance(a, torch.Tensor):
        b = b.to(a.device)
        if a.shape != b.shape:
            import pudb;
            pudb.set_trace()
        if not torch.all(a == b):
            print(f"{(a - b).abs().max():1.2e} {key}")
            return False
        return True
    elif isinstance(a, T.Mapping):
        results = dict()
        for k, v in a.items():
            if k not in b:
                raise KeyError(
                    f"Key error: '{k}'. Similar keys: {difflib.get_close_matches(k, b.keys())}")
            results[k] = compare(v, b[k], key=k)
        result = all(results.values())
        if not result:
            print(str(results).replace(", ", ",\n"))
        return result
    elif isinstance(a, T.Sequence):
        return compare(dict(enumerate(a)), dict(enumerate(b)))
    else:
        if a != b:
            return False
        return True
