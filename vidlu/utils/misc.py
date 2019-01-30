from collections.abc import Mapping
import inspect
from collections import defaultdict
import select
import sys
import platform


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


# Slicing ##########################################################################################

def slice_len(s, sequence_length):
    # stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


# Dictionary-like trees ############################################################################

def tree_to_paths(tree) -> list:
    return [[([k] + p, v) for p, v in tree_to_paths(v)] if isinstance(v, type(tree)) else [([k], v)]
            for k, v in tree.items()]


def copy_tree(tree):
    tree_type = type(tree)
    return tree_type(**{k: v.copy() if isinstance(v, tree_type) else v for k, v in tree.items()})


def paths_to_tree(path_value_pairs, tree_type):
    class Leaf:  # used to encode leaves to distinguish lists from
        def __init__(self, item):
            self.item = item

    subtrees = defaultdict(list)
    for path, value in path_value_pairs:
        if len(path) > 1:
            subtrees[path[0]] += [(path[1:], value)]
        else:
            subtrees[path[0]] = Leaf(value)
    return tree_type({k: v.item if type(v) is Leaf else paths_to_tree(v, tree_type)
                      for k, v in subtrees.items()})


def number_of_leaves(tree: Mapping, tree_type):
    return sum(
        number_of_leaves(v, tree_type) if isinstance(v, tree_type) else 1 for k, v in tree.items())


# Event ############################################################################################

class Event:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)
        return handler  # for usage as decorator

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def __call__(self, *args, **kwargs):
        for h in self.handlers:
            h(*args, **kwargs)


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
