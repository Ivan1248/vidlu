import numpy as np
import torch


def random_sample_from_p_ball(nelem, p):
    # https://mathoverflow.net/questions/9185/how-to-generate-random-points-in-ell-p-balls
    if p == np.inf:
        return torch.empty(nelem).uniform_(-1, 1)
    elif p == 2:
        # arr = torch.distributions.Normal(0, 1/2**2*nelem**(1/p)).rsample((nelem,))
        # z = torch.empty(()).exponential_()
        # return arr.div_((z + arr.pow(2).sum()).sqrt())
        arr = torch.empty(nelem).normal_(0, 1)
        r = torch.empty(()).uniform_(0, 1).pow_(1 / nelem)
        return arr.mul_(r / arr.norm(2))
    elif p == 1:
        arr = torch.distributions.Laplace(0, 1).rsample((nelem,))
        r = torch.empty(()).uniform_(0, 1).pow_(1 / nelem)
        return arr.mul_(r / arr.norm(1))
    elif p == 'inf':
        arr = torch.distributions.Uniform(-1, 1).rsample((nelem,))
        r = torch.empty(()).uniform_(0, 1).pow_(1 / nelem)
        return arr.mul_(r / arr.norm(1))
