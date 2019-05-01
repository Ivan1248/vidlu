from numbers import Real

import numpy as np
import torch
import torch.distributions as D


def generalized_normal_distribution(p, loc, scale):
    if p in [np.inf, 'inf']:
        return D.Uniform(loc - scale, loc + scale)
    if p == 2:
        return D.Normal(loc, scale)
    elif p == 1:
        return D.Laplace(loc, scale)


def generalized_normal_sample(p, loc=0, scale=1, shape=()):
    """Samples from a generalized normal distribution with a 1 degree of
    freedom diagonal covariance matrix.

    p(x) = 1/Z*exp(-(|x-loc|/scale)^p), Z = p/(2*scale*gamma(1/p))

    Args:
        p: the shape parameter of the distribution.
        loc: the location (mean) parameter of the distribution.
        scale: the scale parameter of the distribution.
        shape: the number of elements or the shape of the array.

    Returns:
        A random sample from a generalized normal distribution with a 1 degree
        of freedom diagonal covariance matrix.
    """
    if isinstance(shape, Real):
        shape = (shape,)
    if p in [np.inf, 'inf']:
        return torch.empty(shape).uniform_(loc - scale, loc + scale)
    if p == 2:
        return torch.empty(shape).normal_(loc, scale)
    elif p == 1:
        return D.Laplace(loc, scale).rsample(shape)


def uniform_sample_from_p_ball(p, shape=()):
    """Samples from a uniform distribution over unit a p-ball.

    A sample from a p-generalized normal distribution is taken, divided by its
    p-norm, and multiplied by a random scalar representing the final radius.

    Args:
        p: p-ball p.
        shape: the number of elements or the shape of the array.

    Returns:
        A random sample from a uniform distribution over a unit p-ball.
    """
    if isinstance(shape, Real):
        shape = (shape,)
    arr = generalized_normal_sample(p, 0, 1, shape=shape)
    if p in [np.inf, 'inf']:
        return arr
    r = torch.empty(()).uniform_(0, 1).pow_(1 / np.prod(shape))
    return arr.mul_(r / arr.norm(p))
