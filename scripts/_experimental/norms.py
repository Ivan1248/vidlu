import numpy as np
import torch
import matplotlib.pyplot as plt

plt.figure()

numel_base = 4
numel_max_order_of_magnitude = 6
ps = 2. ** np.arange(-6, 13)
for m in range(numel_max_order_of_magnitude + 1):
    numel = numel_base ** m
    x = torch.ones(numel, dtype=torch.double)
    norms = np.zeros(len(ps))
    sample_count = 4 * int(1 + 2 ** ((12 - m - 1) / 1))
    print(sample_count)
    for _ in range(sample_count):
        x.uniform_(-1, 1).pow_(1 / numel)

        for i, p in enumerate(ps):
            # norms[i] += [(x.norm(p) / (x.size(0) ** (1 / p))).item()]
            norms[i] += [x.abs().pow_(p).mean().pow_(1 / p).item()]
            print(norms[i])
            # norms[i] += [(x.abs().pow_(p).mean().pow_(1 / p)/ (x.size(0) ** (1 / p))).item()]
    for i, p in enumerate(ps):
        norms[i] /= sample_count

    plt.plot(np.log2(ps), norms[:], label=f'n={numel}')

plt.plot(np.log2(ps), (1 / (ps + 1)) ** (1 / ps), label='n=inf')
plt.legend()
plt.show()


def random_sample_from_p_ball(nelem, p):
    # https://mathoverflow.net/questions/9185/how-to-generate-random-points-in-ell-p-balls
    if p == np.inf:
        return torch.empty(nelem).uniform_(-1, 1)
    elif p == 2:
        arr = torch.empty(nelem).normal_(0, 1)
        r = torch.empty(()).uniform_(0, 1).pow_(1 / 2)
        return arr.mul_(r / arr.norm(2))


random_sample_from_p_ball(1000, 2).norm(2)
