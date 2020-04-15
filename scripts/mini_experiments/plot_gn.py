# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.special import gamma

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(-2.5, 2.5, 2e-3, dtype=np.double)
p = 2 ** np.arange(-5, 5, 2e-3, dtype=np.double)
x, p = np.meshgrid(x, p)
z = p / (2 * gamma(1 / p)) * np.exp(-np.abs(x) ** p)

# Plot the surface.
surf = ax.plot_surface(x, p, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
