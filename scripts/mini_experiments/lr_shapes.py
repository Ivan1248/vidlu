import numpy as np
import matplotlib.pyplot as plt


def cosanneal(x, eta_max=1, eta_min=0):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(x * np.pi))


def steppy(x, step_locs=(0.3, 0.6, 0.8), mul=0.2):
    p = sum(x >= l for l in step_locs)
    return mul ** p


x = np.linspace(0, 1, 1000)
yc = cosanneal(x)
ys = steppy(x)

plt.figure()
plt.plot(x, yc)
plt.plot(x, ys)
plt.show()
