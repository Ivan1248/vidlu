from torch import nn

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