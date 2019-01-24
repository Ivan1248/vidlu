import numpy as np
import matplotlib.pyplot as plt


def plot_surface(function, rect, offsets=[0.5], center=0.5, width=256, height=256, axis=None):
    """
    Creates a surface plot (visualize with plt.show)
    From http://www.zemris.fer.hr/~ssegvic/du/

    Arguments:
        function: surface to be plotted
        rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
        offset:   the level plotted as a contour plot

    Returns:
        None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    #get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = center if center else 0.5
    maxval = max(np.max(values) - delta, -(np.min(values) - delta))

    # draw the surface and the offset
    ax = axis or plt
    ax.pcolormesh(
        xx0,
        xx1,
        values,
        vmin=delta - maxval,
        vmax=delta + maxval,
        cmap='RdYlGn')

    if offsets != None:
        ax.contour(xx0, xx1, values, colors='black', levels=offsets)
