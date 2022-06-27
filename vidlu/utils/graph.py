# Taxonomy

class DiGraphBase:
    def edges(self):
        raise NotImplementedError()

    def leaves(self):
        raise NotImplementedError()

    def roots(self):
        return self.reverse().leaves()

    def reverse(self):
        raise NotImplementedError()


class StructuredDAG(dict, DiGraphBase):
    """Keys are classes, and values are TreeTaxonomy instances.
    All nodes are directly accessible at the root level unless
    all_nodes_in_root = False -- then, only root nodes are directly
    accessible."""

    def __init__(self, edges, all_nodes_in_root=True):
        super().__init__()
        self.all_nodes_in_root = all_nodes_in_root
        non_root = set()
        for sup, sub in edges:
            if sup not in self:
                self[sup] = StructuredDAG()
            if sub not in self:
                self[sub] = StructuredDAG()
            self[sup][sub] = self[sub]
            non_root.add(sub)
        if not all_nodes_in_root:
            for k in non_root:
                del self[k]

    def visit(self, func):
        visited = set()
        for k, subs in self.items():
            if k in visited:
                continue
            visited.add(k)
            yield from func(k, subs)
            yield from subs.visit(func)

    def edges(self):
        return self.visit(lambda k, subs: ((k, sub) for sub in subs.keys()))

    def leaves(self):
        return (k for k, subs in self.items() if len(subs) == 0)

    def roots(self):
        return self.keys() if self.all_nodes_in_root else super().roots()

    def reverse(self):
        return StructuredDAG(Edges(self).reverse())

    def nodes(self):
        return self.visit(lambda k, subs: (k, subs))


class Tree(dict, StructuredDAG):
    """Keys are classes, and values are TreeTaxonomy instances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, all_nodes_in_root=False, **kwargs)
        if not self.check():
            raise RuntimeError("The provided edges do not represent a tree.")

    def check(self):
        nodes = list(k for k, subs in self.nodes())
        return len(nodes) == len(set(nodes))

    def visit(self, func):
        for k, subs in self.items():
            yield from func(k, subs)
            yield from subs.visit(func)

    def leaves(self):
        for k, subs in self.items():
            if len(subs) == 0:
                yield k
            else:
                yield from subs.leaves()

    def reverse(self) -> StructuredDAG:
        return StructuredDAG(Edges(self).reverse())


class Edges(list, DiGraphBase):
    """Graph edges."""

    def __init__(self, arg):
        if isinstance(arg, DiGraphBase):
            super().__init__(arg.edges())
        super().__init__(arg)

    def edges(self):
        yield from self

    def leaves(self):
        return DiGraph(self).leaves()

    def reverse(self):
        return Edges((sub, sup) for sup, sub in self)


class DiGraph(dict, DiGraphBase):
    """Mapping from node to set of nodes."""

    def __init__(self, edges):
        super().__init__()
        for sup, sub in edges:
            if sup not in self:
                self[sup] = set()
            self[sup].add(sub)
            if sub not in self:
                self[sub] = set()

    def edges(self):
        yield ((k, sub) for k, subs in self.items() for sub in subs)

    def leaves(self):
        return (k for k, subs in self.items() if len(subs) == 0)

    def reverse(self):
        return type(self)(Edges(self).reverse())


def collapse(dag: DiGraphBase):
    return DiGraph({k: set(dag[k].leaves()) for k in dag.roots()})
