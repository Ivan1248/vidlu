import math


def cosine_lr(p):
    return (1 + math.cos(math.pi * p)) / 2


def linear_warmup(p, period=0.08):
    return p / period if p < period else 1
