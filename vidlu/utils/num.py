import numpy as np


class KleinSum:
    """ An implementation of the higher-order modification of Neumaier improved
    version of Kahan's algorithm for aggregating floats with less error.

    Based on https://en.wikipedia.org/wiki/Kahan_summation_algorithm.

    Example:
        >>> ks = KleinSum()
        >>> s = 0.0
        >>> n = 1_000_000
        >>> numbers = [5 + np.random.randn() for i in range(n)]
        >>> for x in numbers:
        >>>     ks += x
        >>>     s += x
        >>> print(s / n)
        >>> print(ks.get() / n)
        >>> print(np.sum(numbers) / n)
        >>> print(np.mean(numbers))

    Args:
        type: Number type.
    """

    def __init__(self, type=float):
        self.s = type(0.0)
        self.cs = type(0.0)
        self.ccs = type(0.0)

    def get(self):
        return self.s + self.cs + self.ccs

    def __iadd__(self, x):
        s, cs, ccs = self.s, self.cs, self.ccs
        t = s + x
        if np.abs(s) >= np.abs(x):
            c = (s - t) + x
        else:
            c = (x - t) + s
        s = t
        t = cs + c
        if np.abs(cs) >= np.abs(c):
            cc = (cs - t) + c
        else:
            cc = (c - t) + cs
        self.s, self.cs, self.ccs = s, t, ccs + cc
        return self

    def __repr__(self):
        return f"KleinSum({self.get()})"


def round_to_int(x):
    return (x + 0.5).astype(int)
