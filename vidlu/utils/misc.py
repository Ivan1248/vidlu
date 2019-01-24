from collections.abc import Mapping
import inspect
import select
import sys
import platform

# Empty type - representing an unassigned variable ###################################################

Empty = inspect.Parameter.empty


def is_none_or_empty(value):
    return value is None or value is Empty


# Inspection #######################################################################################

def locals_from_first_initializer():
    """
    Returns arguments of the constructor of a subclass if `super().__init__()`
    is the first statement in the subclass' `__init__`. Taken from MagNet
    (https://github.com/MagNet-DL/magnet/blob/master/magnet/utils/misc.py) and
    modified. Originally `magnet.utils.misc.caller_locals`.
    """
    frame = inspect.currentframe().f_back.f_back

    try:
        l = frame.f_locals

        f_class = l.pop('__class__', None)
        caller = l.pop('self')
        while f_class is not None and isinstance(caller, f_class):
            l.pop('args', None)
            args = frame.f_locals.pop('args', None)
            l.update(frame.f_locals)
            if args is not None:
                l['args'] = args

            l.pop('self', None)
            frame = frame.f_back
            f_class = frame.f_locals.pop('__class__', None)

        l.pop('self', None)
        l.pop('__class__', None)
        return l
    finally:
        del frame


# Console ##########################################################################################

def try_get_input(impatient=False):
    """
    Returns a line from standard input.
    :param impatient: If set to True and no input is available, None is returned
        instead of waiting for input.
    """

    def input_available():
        if platform.system() == 'Windows':
            import msvcrt
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0)[0]

    if impatient and not input_available():
        return None
    return input()


# Slicing ##########################################################################################

def slice_len(s, sequence_length):
    # stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


# Dictionary trees #################################################################################

def tree_to_paths(tree: Mapping) -> list:
    paths = []
    for k, v in tree.items():
        paths += [([k] + p, v) for p, v in tree_to_paths(v)] if isinstance(v, Mapping) \
            else [([k], v)]
    return paths


def number_of_leaves(tree: Mapping):
    n = 0
    for k, v in tree.items():
        n += number_of_leaves(v) if isinstance(v, Mapping) else 1
    return n


# Dict map, filter #################################################################################

def dict_map(func, dict_):
    return {k: func(v) for k, v in dict_.items()}


def dict_filter(func, dict_):
    return {k: v for k, v in dict_.items() if func(v)}
