"""Learning rate shapes: functions mapping training progress `p` in `[0, 1]` to a
multiplicative learning rate factor.

They are pure and unvalidated. `vidlu.optim.lr_schedulers.ScalableLR` computes the progress
and is what rejects a value outside `[0, 1]`.
"""

import math


def cosine_lr(p):
    return (1 + math.cos(math.pi * p)) / 2


def quarter_cos(p):
    return math.cos(p * math.pi / 2)


def linear_warmup(p, period=0.08):
    return p / period if p < period else 1


def poly(p, power=0.9):  # power=0.9 makes it very close to the "ramp" shape
    return (1 - p) ** power


def ramp(p):
    return 1 - p


def with_warmup(shape, warmup_proportion=0.1, start_factor=0.1, min_factor=0.):
    """Returns a shape that linearly warms up to 1 and then follows `shape` down to `min_factor`.

    Warmup goes from `start_factor` to 1 over the progress interval `[0, warmup_proportion]`.
    The remaining progress is rescaled to `[0, 1]` and passed to `shape`, whose result is
    mapped from `[0, 1]` onto `[min_factor, 1]`.

    Args:
        shape: A learning rate shape mapping progress in `[0, 1]` to a factor in `[0, 1]`.
        warmup_proportion: Proportion of the progress interval spent warming up, in `[0, 1)`.
        start_factor: Factor at progress 0.
        min_factor: Factor at progress 1.

    Returns:
        A function mapping progress in `[0, 1]` to a factor.

    Raises:
        ValueError: If an argument is outside its allowed range.
    """
    if not 0 <= warmup_proportion < 1:
        raise ValueError(f"{warmup_proportion=} should be in [0, 1).")
    if not 0 <= start_factor <= 1:
        raise ValueError(f"{start_factor=} should be between 0 and 1.")
    if not 0 <= min_factor <= 1:
        raise ValueError(f"{min_factor=} should be between 0 and 1.")

    def warmed_up_shape(p):
        if p < warmup_proportion:
            return start_factor + (1 - start_factor) * linear_warmup(p, period=warmup_proportion)
        decay_progress = (p - warmup_proportion) / (1 - warmup_proportion)
        return min_factor + (1 - min_factor) * shape(decay_progress)

    return warmed_up_shape
