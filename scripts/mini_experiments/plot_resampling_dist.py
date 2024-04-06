import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

N = 200
colors = pl.cm.jet(np.linspace(0, 1, N))
for i in range(N):
    plt.plot(np.linspace(0, 1, k := int(1.05 ** i + 0.5)),
             np.sort(np.unique(np.random.randint([k] * (k * 100)), return_counts=True)[1])[::-1],
             alpha=0.05, c=colors[i])
for i in range(N):
    plt.plot(np.linspace(0, 1, k := int(1.05 ** i + 0.5)),
             np.sort(np.unique(np.random.randint([k] * (k * 200)), return_counts=True)[1])[::-1],
             alpha=0.05, c=colors[i])
plt.show()
for i in range(N):
    plt.plot(np.linspace(0, 1, k := int(1.05 ** i + 0.5) * 0 + 1000),
             np.sort(np.unique(np.random.randint([1000] * (k * 1000)), return_counts=True)[1])[
             ::-1], alpha=0.05, c=colors[i])
plt.show()
