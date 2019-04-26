import inspect


def class_initializer_locals_c():
    """
    Returns arguments of the initializer of a subclass if there are no variables
    defined before `super().__init__()` calls in the initializer of the object's
    class.
    Based on `magnet.utils.misc.caller_locals`.
    """
    frame = inspect.currentframe().f_back.f_back

    try:
        locals_ = frame.f_locals
        caller = locals_.pop('self')

        while True:
            frame = frame.f_back
            f_class = frame.f_locals.pop('__class__', None)
            locals_curr = frame.f_locals
            if f_class is None or locals_curr.pop('self', None) is not caller:
                break
            locals_ = locals_curr

        locals_.pop('self', None)
        locals_.pop('__class__', None)
        return locals_
    finally:
        del frame  # to avoid cyclical references, TODO


def locals_c(exclusions=('self', '__class__')):
    """ Locals without `self` and `__class__`. """
    frame = inspect.currentframe().f_back
    try:
        locals_ = frame.f_locals
        for x in exclusions:
            locals_.pop(x, None)
        return locals_
    finally:
        del frame  # to avoid cyclical references, TODO


def find_frame_in_call_stack(frame_predicate, start_frame=-1):
    frame = start_frame or inspect.currentframe().f_back.f_back
    while not frame_predicate(frame):
        frame = frame.f_back
        if frame is None:
            return None
    return frame