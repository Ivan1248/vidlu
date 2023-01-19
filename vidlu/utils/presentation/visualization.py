import os
import typing as T
from functools import lru_cache

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm


def concatenate_cmaps(cmaps):
    def conc(x):
        x *= len(cmaps)
        i = int(x)
        if i == x:
            i -= 1
        r = x - i
        return cmaps[i](r)

    return conc


def get_cmap(cmap):
    if isinstance(cmap, str):
        return mpl.cm.get_cmap(cmap)
    elif isinstance(cmap, T.Sequence):
        return concatenate_cmaps(list(map(mpl.cm.get_cmap, cmap)))
    else:
        return cmap


def get_color_palette(n, cmap='jet'):
    cmap = get_cmap(cmap)
    return np.stack([np.array(cmap(i / (n - 1))[:3]) for i in range(n)])


def fuse_images(im1, im2, a=0.5):
    return a * im1 + (1 - a) * im2


def scale_min_max(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def colorize_segmentation(seg, colors):
    return colors[seg.ravel()].reshape(seg.shape + colors.shape[1:])


####

def show_batch(x, nrows=None, ncols=None):
    if x.shape[1] not in (1, 3):
        if nrows is None:
            nrows = len(x)
        x = x.view(-1, *x.shape[2:])
    from torchvision.utils import make_grid
    grid = make_grid(x, nrow=nrows or (len(x) // (ncols or int(len(x) ** 0.5))))
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.show()


def show_segmentation_prediction(y, x=None):
    nrows = class_count = y.shape[1]
    y = y.view(-1, 1, *y.shape[2:]).repeat_interleave(3, dim=1)
    if x is not None:
        x = x.repeat_interleave(class_count, dim=0)
        y *= x
        del x
    return show_batch(y, nrows=nrows)


def composef(images, fmt):
    """Composes images into a grid according to the format.

    Args:
        images:
        fmt:

    Example:
        >>> format = lambda x: [[x[0], None], [fuse_images(x[0],x[1]), x[1]]]
    """
    images_array = fmt(images)
    return compose(images_array)


def compose(images_array):
    import torch  # np.concatenate somtimes gets stuck for some reason
    if not isinstance(images_array[0], list):
        images_array = [images_array]
    rows = [torch.cat(list(map(torch.from_numpy, row)), dim=1) for row in images_array]
    return torch.cat(rows, axis=0).numpy()


class Viewer:
    """Datasset viewer.

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


def normalize_colors(colors, insert_zeros=False):
    colors = list(map(np.array, colors))
    if np.max(np.array(colors)) > 1:
        colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    return np.array([np.zeros(3)] + colors) if insert_zeros else np.array(colors)


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
                return colorize_segmentation(pred + 1, colors)

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


def view_predictions(dataset, infer=None, save_dir=None, colors=None, class_count=None):
    if colors is not None:
        colors = [np.zeros(3)] + colors  # unknown black
    elif class_count is not None:
        colors = get_color_palette(class_count)
    elif 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = np.array([np.zeros(3)] + colors)  # unknown black


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
                pred_disp = colorize_segmentation(pred_disp + 1, colors)

            def _get_class_representative():
                cr = get_class_representative(int(pred))
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

        bar_width, bar_height = comp.shape[1] // 20, comp.shape[0]
        step = bar_height // len(colors)
        bar = np.zeros((bar_height, bar_width), dtype=np.int8)
        for i in range(len(colors)):
            bar[i * step:(i + 1) * step, 1:] = len(colors) - 1 - i
        bar = colorize_segmentation(bar, colors)

        return compose([comp, bar])

    if save_dir is not None:
        from PIL import Image
        from tqdm import tqdm
        print("Saving predictions")

        os.makedirs(save_dir, exist_ok=True)
        for i, d in enumerate(tqdm(dataset)):
            im = np.round(get_frame(d) * 255).astype('uint8')
            Image.fromarray(im).save(f'{save_dir}/p{i:05d}.png')

    return Viewer().display(dataset, get_frame)


def generate_adv_iter_segmentations(dataset, model, attack, save_dir):
    if 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = np.array([np.zeros(3)] + colors)  # unknown black

    def get_frame(img, lab, img_adv, pred):
        fuse = lambda im, seg: fuse_images(im, colorize_segmentation(seg + 1, colors), 0.2)
        return compose([[img, img_adv], [fuse(img_adv, lab), fuse(img_adv, pred)]])

    from PIL import Image
    print("Saving predictions")

    os.makedirs(save_dir, exist_ok=True)
    for i, d in enumerate(tqdm(dataset)):
        dir = f'{save_dir}/p{i:05d}'
        os.makedirs(dir, exist_ok=True)

        def save_frame(s):
            import torch
            with torch.no_grad():
                args = [s.x.permute(0, 2, 3, 1), s.y_adv, s.x_adv.permute(0, 2, 3, 1),
                        s.out.argmax(1)]
                args = [a[0].detach().cpu().numpy() for a in args]
                x = get_frame(*args)
                im = np.round(x * 255).astype('uint8')
                Image.fromarray(im).save(f'{dir}/{s.step:05d}.png')

        attack.perturb(model, d.x.unsqueeze(0).cuda(), d.y.unsqueeze(0).cuda(),
                       backward_callback=save_frame)


def plot_curves(curves, xlim=None, ylim=None, xticks=None, yticks=None):
    fig = plt.figure()
    axes = plt.gca()
    if yticks:
        plt.yticks(yticks)
    if xticks:
        plt.xticks(xticks)
    if xlim:
        axes.set_ylim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    axes.grid(color='0.9', linestyle='-', linewidth=1)
    for name, (x, y) in curves.items():
        plt.plot(x, y, label=name, linewidth=1)
    plt.xlabel("broj završenih epoha")
    plt.legend()
    return fig, axes
