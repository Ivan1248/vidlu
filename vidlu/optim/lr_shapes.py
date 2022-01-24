import math


def cosine_lr(p):
    return (1 + math.cos(math.pi * p)) / 2


def linear_warmup(p, period=0.08):
    return p / period if p < period else 1


def poly(p, power=0.9):  # power=0.9 makes it very close to the "ramp" shape
    return (1 - p) ** power


def ramp(p):
    return 1 - p
