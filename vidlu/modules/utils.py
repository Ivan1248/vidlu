from torch import nn
from inspect import isfunction
from functools import wraps
import contextlib

from vidlu.utils.inspect import find_frame_in_call_stack


def get_calling_module(module, start_frame=None):
    def predicate(frame):
        locals_ = frame.f_locals
        if 'self' in locals_:
            self_ = locals_['self']
            return isinstance(locals_['self'], nn.Module) and self_ is not module

    frame = find_frame_in_call_stack(predicate, start_frame)
    if frame is None:
        return None, None
    caller = frame.f_locals['self']
    return caller, frame


def try_get_module_name_from_call_stack(module, start_frame=None, full_name=True):
    parent, frame = get_calling_module(module, start_frame=start_frame)
    if parent is None:
        return 'ROOT'
    for n, c in parent.named_children():
        if c is module:
            if not full_name:
                return n
            parent_name = try_get_module_name_from_call_stack(parent, start_frame=frame.f_back)
            return f'{parent_name}.{n}'
    return '???'


def get_device(module):
    param = next(module.parameters(), None)
    return None if param is None else param[0].device


def eval_no_param_grad(*modules):
    r"""A Context Manger that sets modules to eval mode and disables gradient
    computation only for parameters.
    Based on
    https://github.com/MagNet-DL/magnet/blob/master/magnet/_autograd.py.

    It sets modules in their ``eval`` mode and ensures that gradients with
    respect to parameters are not computed.
    This is a more wholesome option than :py:meth:`torch.no_grad` since many
    modules (BatchNorm, Dropout etc.) behave differently while training and
    testing.
    Examples::
    >>> import vidlu
    >>> import vidlu.modules as vm
    >>> import torch
    >>> model = vm.Linear(10)
    >>> x = torch.randn(4, 3)
    >>> with vm.eval_no_param_grad(model):  # Using eval() as context manager
    >>>     model(x)
    >>>
    >>> @vm.eval_no_param_grad(model)  # Using as decorator
    >>> def foo():
    >>>     return model(x)
    >>> foo()
    >>>
    >>> # The modules can also be given at runtime by specifying no arguments
    >>> @vm.eval
    >>> def foo(model):
    >>>     return model(x)
    >>> foo()
    >>> # The method then takes modules from the arguments
    >>> # to the decorated function.
    """

    @contextlib.contextmanager
    def eval_no_param_grad_cm(*modules):
        modules = [module for module in modules if module.training]
        mstates = [module.training for module in modules]
        params = [p for module in modules if module.training for p in module.parameters()]
        pstates = [p.requires_grad for p in params]
        for module in modules:
            module.eval()
        for p in params:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for module, state in zip(modules, mstates):
                module.train(state)
            for p, pstate in zip(params, pstates):
                p.requires_grad_(pstate)

    # Check if called as decorator
    if not isfunction(modules[0]) or len(modules) > 1:
        return eval_no_param_grad_cm(*modules)

    fn = modules[0]  # The decorated function

    @wraps(fn)
    def new_fn(*args, **kwargs):
        arg_list = list(args) + list(kwargs.values())
        modules = [a for a in arg_list if isinstance(a, nn.Module)]

        with eval_no_param_grad_cm(*modules):
            return fn(*args, **kwargs)

    return new_fn
