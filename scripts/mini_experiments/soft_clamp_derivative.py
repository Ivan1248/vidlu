# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import torch

import vidlu.ops as vo

fig = plt.figure()

# Make data.
x = torch.linspace(-2, 3, 4000, requires_grad=True)
ylim = torch.clamp(x, 0, 1)
y1 = torch.tanh(x)
yst = torch.tanh(x * 2 - 1) / 2 + 0.5
plt.plot(x.detach().numpy(), y1.detach().numpy(), label='tanh')

for k in reversed([1, 2, 4, 8, 16, 32, 64, 128]):
    y = vo.soft_clamp(x, 0, 1,  1 / (2 * k))
    x.grad = None
    y.sum().backward()
    plt.plot(x.detach().numpy(), y.detach().numpy(), label=k)
    plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label=f'{k}_grad')

plt.legend()
plt.show()
