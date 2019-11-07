from _context import vidlu
import numpy as np
import torch
import matplotlib.pyplot as plt

from vidlu.ops.random import uniform_sample_from_p_ball

maxi = 9
for i in range(maxi):
    a, b = [torch.stack([uniform_sample_from_p_ball(p, (10 ** i))
                         for _ in range(5 ** (maxi - i - 1))]).view(-1)
            for p in [np.inf, 'inf']]
    plt.figure()
    plt.hist(a, alpha=0.5)
    plt.hist(b, alpha=0.5)
    print(a, b)
    plt.show()
