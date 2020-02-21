from collections import defaultdict
import typing as T


# Dictionary-like trees ############################################################################

def equals(a, b, tree_type=None, ignore_order=True):
    tree_type = tree_type or type(a)
    return ((ignore_order or a.keys() == b.keys())
            and all(equals(a[k], b[k], tree_type, ignore_order) if isinstance(a[k], tree_type)
                    else a[k] == b[k] for k in a))


def copy(tree, tree_type=None):
    tree_type = tree_type or type(tree)
    return tree_type(**{k: copy(v, tree_type) if isinstance(v, tree_type) else v
                        for k, v in tree.items()})


def flatten(tree, tree_type=None) -> list:
    tree_type = tree_type or type(tree)

    out = []
    for k, v in tree.items():
        if isinstance(v, tree_type) and len(v) != 0:
            out.extend(((k,) + p, v) for p, v in flatten(v))
        else:
            out.append(((k,), v))
    return out


def unflatten(path_to_value: T.Union[T.Iterable[tuple], T.Mapping], tree_type=dict):
    class Leaf:  # used to encode leaves to distinguish lists from
        def __init__(self, item):
            self.item = item

    subtrees = defaultdict(list)
    if isinstance(path_to_value, T.Mapping):
        path_to_value = path_to_value.items()
    for path, value in path_to_value:
        if len(path) > 1:
            subtrees[path[0]] += [(path[1:], value)]
        else:
            subtrees[path[0]] = Leaf(value)
    return tree_type(**dict([(k, v.item if type(v) is Leaf else unflatten(v, tree_type))
                             for k, v in subtrees.items()]))


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


def convert(tree, out_tree_type, in_tree_type=None, convert_empty_trees=True,
            recurse_sequences=False):
    in_tree_type = in_tree_type or type(tree)

    def recurse(v):
        if isinstance(v, in_tree_type) and (convert_empty_trees or len(v) > 0):
            return convert(v, out_tree_type, in_tree_type, convert_empty_trees, recurse_sequences)
        elif recurse_sequences and isinstance(v, (list, tuple)):
            return type(v)(recurse(u) for u in v)
        return v

    return out_tree_type(**{k: recurse(v) for k, v in tree.items()})


def map(tree, func, tree_type=None):
    tree_type = tree_type or type(tree)
    return tree_type(**{k: map(v, func, tree_type) if isinstance(v, tree_type) else func(v)
                        for k, v in tree.items()})


def to_dot(tree, label="ROOT", graph=None, tree_type=None, max_label_length=999999999):
    tree_type = tree_type or type(tree)
    import pydot

    def elipsis(s):
        return (s[:max_label_length] + '..') if len(s) > max_label_length else s

    graph = graph or pydot.Dot(graph_type='digraph', rankdir='LR')
    graph.add_node(pydot.Node(id(tree), label=elipsis(f"{label}"), shape="box"))
    for k, v in tree.items():
        if isinstance(v, tree_type):
            to_dot(v, label=k, graph=graph, parent=tree, tree_type=tree_type)
            graph.add_edge(pydot.Edge(id(tree), id(v)))
        else:
            id_ = hash((id(tree), id(k)))
            graph.add_node(pydot.Node(id_, label=elipsis(f"{k}={v}"), shape="box"))
            graph.add_edge(pydot.Edge(id(tree), id_))
    return graph
