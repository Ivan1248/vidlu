import os
from functools import lru_cache

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_color_palette(n, cmap='jet'):
    cmap = mpl.cm.get_cmap(cmap)
    return [np.array(cmap(i / (n - 1))[:3]) for i in range(n)]


def fuse_images(im1, im2, a=0.5):
    return a * im1 + (1 - a) * im2


def scale_min_max(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def colorify_segmentation(seg, colors):
    plab = np.empty(list(seg.shape) + [3])
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            plab[i, j, :] = colors[seg[i, j]]
    return plab


def composef(images, fmt):
    """

    Args:
        images:
        fmt:

    Example:
        >>> format = lambda x: [[x[0], None], [fuse_images(x[0],x[1]), x[1]]]
    """
    images_array = fmt(images)
    return compose(images_array)


def compose(images_array):
    if not isinstance(images_array[0], list):
        images_array = [images_array]

    rows = [np.concatenate(row, axis=1) for row in images_array]
    return np.concatenate(rows, axis=0)


def compose_old(images, fmt='0,0;1,0-1'):
    if fmt is None:
        return np.concatenate(
            [np.concatenate([im for im in row], 1) for row in images], 0)

    def get_image(frc):
        inds = [int(i) for i in frc.split('-')]
        assert (len(inds) <= 2)
        ims = [images[i] for i in inds]
        return ims[0] if len(ims) == 1 else fuse_images(ims[0], ims[1], 0.5)

    fmt = fmt.split(';')
    fmt = [f.split(',') for f in fmt]
    return np.concatenate([
        np.concatenate([get_image(frc) for frc in frow], 1) for frow in fmt
    ], 0)


class Viewer:
    """
        Press "q" to close the window. Press anything else to change the displayed
        composite image. Press "a" to return to the previous image.
    """

    def __init__(self, name='Viewer'):
        self.name = name

    def display(self, dataset, mapping=lambda x: x):
        # mpl.use('wxAgg')

        i = 0

        def get_images(i):
            images = mapping(dataset[i])
            return images if isinstance(images, list) else [images]

        def show(i):
            images = get_images(i)
            for axim, im in zip(aximgs, images):
                axim.set_data(im)
            fig.canvas.set_window_title(str(i) + "-" + self.name)
            fig.canvas.draw()

        def on_press(event):
            nonlocal i
            if event.key == 'left':
                i -= 1
            elif event.key == 'right':
                i += 1
            elif event.key == 'q' or event.key == 'esc':
                plt.close(event.canvas.figure)
                return
            i = i % len(dataset)
            show(i)

        images = get_images(0)
        subplot_count = len(images)

        nrows = int(subplot_count ** 0.5)
        ncols = int(subplot_count // nrows + 0.5)

        fig, axes = plt.subplots(nrows, ncols)
        if subplot_count == 1:
            axes = [axes]
        else:
            axes = axes.flat[:subplot_count]
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.set_window_title(self.name)

        def make_valid(im):
            if np.min(im) < 0 or np.max(im) > 0:
                return scale_min_max(im)
            return im

        plot = lambda ax, im: ax.imshow(make_valid(im)) if len(im.shape) == 3 else ax.imshow(im)
        aximgs = [plot(ax, im) for ax, im in zip(axes, images)]
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for ax, axim in zip(axes, aximgs):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(axim, cax=cax, orientation='vertical')
        plt.show()
        show(0)


def view_predictions_2(dataset, infer=None, save_dir=None):
    if 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = [np.zeros(3)] + list(map(np.array, colors))  # unknown black

    def get_frame(datapoint):
        def process(pred):
            if np.issubdtype(pred.dtype, np.floating):
                return pred
            else:
                return colorify_segmentation(pred + 1, colors)

        img, lab = datapoint
        return [scale_min_max(img)] + list(map(process, [lab] + list(infer(img))))

    if save_dir is not None:
        from skimage.io import imsave
        from tqdm import tqdm
        print("Saving predictions")

        os.makedirs(save_dir, exist_ok=True)
        for i, d in enumerate(tqdm(dataset)):
            imsave(f'{save_dir}/p{i}.png', get_frame(d))

    return Viewer().display(dataset, get_frame)


def view_predictions(dataset, infer=None, save_dir=None):
    if 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = [np.zeros(3)] + list(map(np.array, colors))  # unknown black

    @lru_cache(maxsize=1000)
    def get_class_representative(label):  # classification
        for x, y in dataset:
            if y == label:
                return x

    def get_frame(datapoint):
        img, lab = datapoint
        classification = np.shape(lab) == ()
        img_scal = scale_min_max(img)
        black = img_scal * 0
        comp_arr = [[img_scal, black]]

        def add_prediction(pred):
            pred_disp = np.full(img.shape[:2], pred) if classification else pred
            if np.issubdtype(pred_disp.dtype, np.floating):
                shape = list(pred_disp.shape) + [3]
                pred_disp = np.repeat(pred_disp, 3, axis=-1)
                pred_disp = np.reshape(pred_disp, shape)
                pred_disp = scale_min_max(pred_disp)
            else:
                pred_disp = colorify_segmentation(pred_disp + 1, colors)

            def _get_class_representative():
                cr = get_class_representative(pred)
                return black if cr is None else scale_min_max(cr)

            pred_img = (_get_class_representative() if classification
                        else fuse_images(img_scal, pred_disp))
            comp_arr.append([pred_img, pred_disp])

        add_prediction(lab)

        if infer is not None:
            preds = infer(img)
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
            for pred in preds:
                add_prediction(pred)

        comp = compose(comp_arr)

        bar_width, bar_height = comp.shape[1] // 10, comp.shape[0]
        step = bar_height // len(colors)
        bar = np.zeros((bar_height, bar_width), dtype=np.int8)
        for i in range(len(colors)):
            bar[i * step:(i + 1) * step, 1:] = len(colors) - 1 - i
        bar = colorify_segmentation(bar, colors)

        return compose([comp, bar])

    if save_dir is not None:
        from skimage.io import imsave
        from tqdm import tqdm
        print("Saving predictions")

        os.makedirs(save_dir, exist_ok=True)
        for i, d in enumerate(tqdm(dataset)):
            imsave(f'{save_dir}/p{i:05d}.png', get_frame(d))

    return Viewer().display(dataset, get_frame)


def plot_curves(curves):
    # plt.yticks(np.arange(0, 0.51, 0.05))
    # axes.set_xlim([0, 200])
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.grid(color='0.9', linestyle='-', linewidth=1)
    for name, (x, y) in curves.items():
        plt.plot(x, y, label=name, linewidth=1)
    plt.xlabel("broj završenih epoha")
    plt.legend()
    plt.show()
