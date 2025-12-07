import contextlib


@contextlib.contextmanager
def switch_var(getter, setter, value, *, omit_unnecessary_calls=False):
    state = getter()
    switch_necessary = not omit_unnecessary_calls or (state == value)
    if switch_necessary:
        setter(value)
    yield
    if switch_necessary:
        setter(state)


@contextlib.contextmanager
def preserve_var(getter, setter):
    state = getter()
    yield
    setter(state)


@contextlib.contextmanager
def switch_attribute(objects, attrib_name, value):
    state = {m: getattr(m, attrib_name) for m in objects}
    for m in state:
        setattr(m, attrib_name, value)
    yield
    for m, v in state.items():
        setattr(m, attrib_name, v)


@contextlib.contextmanager
def switch_attributes(objects, **name_to_value):
    with contextlib.ExitStack() as stack:
        for name, value in name_to_value.items():
            stack.enter_context(switch_attribute(objects, name, value))
        yield


def switch_attribute_if_exists(objects, attrib_name, value):
    return switch_attribute((k for k in objects if hasattr(k, attrib_name)), attrib_name, value)


class _ClassAttr:
    pass


@contextlib.contextmanager
def preserve_attribute(objects, attrib_name, copy_func=lambda x: x):
    state = {obj: copy_func(getattr(obj, attrib_name)) 
             if attrib_name in vars(obj) else _ClassAttr
             for obj in objects}
    yield
    for obj, v in state.items():
        if v is _ClassAttr:
            if attrib_name in vars(obj):
                del vars(obj)[attrib_name]
        else:
            setattr(obj, attrib_name, v)
