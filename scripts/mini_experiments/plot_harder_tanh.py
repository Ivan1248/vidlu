# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.special import gamma


def soft_max(x, y, h=1):
    return np.log(np.exp(x * h) + np.exp(y * h)) / h


def soft_min(x, y, h=1):
    return np.log(np.exp(x * -h) + np.exp(y * -h)) / -h
    # return soft_max(x, y, -h)


def soft_clamp(x, min_, max_, h=1):
    elh, euh = np.exp(min_ * h), np.exp(max_ * h)
    c = (max_ + min_) / 2
    ech = np.exp(c * h)
    s = ech / (ech + elh) * (1 - (ech + elh) / (ech + euh))
    xs = ((x + c * (s - 1)) / s)
    exsh = np.exp(xs * h)
    nom, den = exsh + elh, exsh + euh
    return np.log(nom / den) / h + max_


def harder_tanh(x, h=1):
    # unstable for large h or x
    if h == 0:
        return np.tanh(x)
    euh, elh = np.exp(h), np.exp(-h)
    s = (1 + euh) * (1 + elh) / (euh - elh)  # to make f'(0) = 1
    hs = h * s
    exhs = np.exp(x * hs)
    nom, den = exhs + elh, exhs + euh
    y = (np.log(nom / den)) / h + 1
    y[np.isinf(exhs)] = 1
    return y


def soft_clamp(x, eps, min_=-2, max_=0.5):
    y = x.copy()
    y[x > max_ - eps] = np.tanh((x[x > max_ - eps] - (max_ - eps)) / eps) * eps + (max_ - eps)
    y[x < min_ + eps] = np.tanh((x[x < min_ + eps] - (min_ + eps)) / eps) * eps + (min_ + eps)
    return y


fig = plt.figure()

# Make data.
x = np.linspace(-2, 3, 4000, dtype=np.float32)
ylim = np.maximum(np.minimum(x, 1), 0)
y1 = np.tanh(x)
yst = np.tanh(x * 2 - 1) / 2 + 0.5
plt.plot(x, y1, label='tanh')

plt.plot(x, ylim, label='lim')
for k in reversed([1, 2, 4, 8, 16, 32, 64, 128]):
    # y = -np.log(np.exp(-k * x) + np.exp(-k)) / k
    # y = 1 / (np.exp(-k * x) + np.exp(-k)) / k * np.exp(-k * x) * k
    # y = soft_max(x, 0, k)
    # y = soft_min(x, 1, k)
    # y = harder_tanh(2 * x - 1, h=k) / 2 + 0.5
    y = soft_clamp(x, 1 / (2 * k), 0, 1)
    # y = soft_clamp(x, eps=5 / 100)
    # y = y * y1 ** 2 + y1 * (1 - y1 ** 2)
    # y = (x * np.exp(-k * x) + np.exp(-k)) / (np.exp(-k * x) + np.exp(-k))
    # y = y1 * y + (1 - y1) * y1
    plt.plot(x, y, label=k)

plt.legend()
plt.show()
