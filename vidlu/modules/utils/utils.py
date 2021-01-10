import itertools
import typing as T

import torch
from torch import nn
from vidlu.utils.func import partial

from vidlu.utils.inspect import find_frame_in_call_stack
from vidlu.utils.func import func_to_class


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


func_to_module_class = partial(func_to_class, superclasses=nn.Module, method_name='forward')


def sole_tuple_to_varargs(inputs):  # tuple, not any sequence type
    return inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs


def extract_tensors(*args, **kwargs):
    for a in itertools.chain(args, kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, T.Sequence):
            for x in extract_tensors(*a):
                yield x
        elif isinstance(a, T.Mapping):
            for x in extract_tensors(*a.values()):
                yield x


def map_tensors(a, f):
    if isinstance(a, torch.Tensor):
        return f(a)
    elif isinstance(a, T.Sequence):
        return type(a)(map_tensors(x, f) for x in a)
    elif isinstance(a, T.Mapping):
        return type(a)({k: map_tensors(x, f) for k, x in a.items()})
