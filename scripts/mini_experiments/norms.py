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