# Experimental. TODO
"""
Encoding:
    union - list
    intersection - tuple
    set of superset-subset relations - vidlu.utils.graph.DiGraph

    taxonomy: [set of classes], []
"""

_Union = list
_Intersection = tuple


def subclasses(a, taxonomy):
    if isinstance(a, _Intersection):
        yield a
    else:
        yield from taxonomy(a)


def normalize(a):
    while len(result) == 1:
        result = result[0]
    return [] if len(result) == 0 else result


def intersection(a, b, taxonomy):
    if a == b:
        return a
    elif len(taxonomy[a]) == 0 or len(taxonomy[b]) == 0:
        return ()
    else:
        return normalize(tuple(sum((intersection(x, y, taxonomy)
                                    for x in subclasses(a, taxonomy)
                                    for y in subclasses(b, taxonomy)), ())))


def union(a, b, taxonomy):
    return normalize(list(subclasses(a, taxonomy)) + list(subclasses(b, taxonomy)))


def intersection_flat(a, b):
    return frozenset({x for x in a if x in b})
