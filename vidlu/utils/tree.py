from collections import defaultdict
from typing import Mapping, Sequence


# Dictionary-like trees ############################################################################


def tree_to_paths(tree, tree_type=None) -> list:
    tree_type = tree_type or type(tree)
    return list(sum(([([k] + p, v) for p, v in tree_to_paths(v)] if isinstance(v, tree_type)
                     else [([k], v)]
                     for k, v in tree.items()), []))


def copy_tree(tree, tree_type=None):
    tree_type = tree_type or type(tree)
    return tree_type(**{k: copy_tree(v, tree_type) if isinstance(v, tree_type) else v
                        for k, v in tree.items()})


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


def number_of_leaves(tree, tree_type=None):
    tree_type = tree_type or type(tree)
    return sum(
        number_of_leaves(v, tree_type) if isinstance(v, tree_type) else 1 for k, v in tree.items())


def print_tree(tree, tree_type=None, depth=0, indent="  "):
    tree_type = tree_type or type(tree)
    for k, v in tree.items():
        if isinstance(v, type(tree)):
            line = f"{indent * depth}{k}: "
            print(line)
            if not indent:
                print_tree(v, tree_type, depth, " " * len(line))
            else:
                print_tree(v, tree_type, depth + 1, indent)
        else:
            print(f"{indent * depth}{k}={repr(v)}")


def convert_tree(tree, output_tree_type, input_tree_type=None, recurse_sequences=False):
    input_tree_type = input_tree_type or type(tree)

    def recurse(v):
        if isinstance(v, input_tree_type):
            return convert_tree(v, output_tree_type, input_tree_type, recurse_sequences)
        elif recurse_sequences and isinstance(v, (list, tuple)):
            return type(v)(recurse(u) for u in v)
        return v

    return output_tree_type(**{k: recurse(v) for k, v in tree.items()})
