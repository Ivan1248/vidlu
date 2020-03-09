from functools import partial
from numbers import Number
import typing as T

from torch import Tensor
import torch.nn.functional as nnF
import numpy as np

from vidlu.utils import num
from vidlu.utils.func import make_multiinput
from vidlu.torch_utils import is_int_tensor, round_float_to_int
from vidlu.modules.utils import func_to_module_class


@make_multiinput
def resize(x, shape=None, scale_factor=None, mode='nearest', align_corners=None):
    additional_dims = (None,) * (4 - len(x.shape))  # must be tuple, empty list doesn't work
    return nnF.interpolate(x[additional_dims], size=shape, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners)[(0,) * len(additional_dims)]


Resize = func_to_module_class(resize)


def resize_segmentation(x, shape=None, scale_factor=None, align_corners=None):
    if not is_int_tensor(x):
        raise TypeError("`x` should be an integer tensor.")
    xfr = resize(x.float(), shape=shape, scale_factor=scale_factor, mode='nearest',
                 align_corners=align_corners)
    return round_float_to_int(xfr, x.dtype)


ResizeSegmentation = func_to_module_class(resize_segmentation)


@make_multiinput
def pad(x, padding, mode='constant', value=0):
    if isinstance(padding, int):
        padding = (padding,) * 4
    additional_dims = (None,) * (4 - len(x.shape))
    return nnF.pad(x[additional_dims], padding, mode=mode, value=value)[(0,) * len(additional_dims)]


Pad = func_to_module_class(pad)


@make_multiinput
def pad_to_shape(x, shape, mode='constant', value=0):
    padding = np.array(shape) - np.array(x.shape[-2:])
    if np.all(padding == 0):
        return x
    elif np.any(padding < 0):
        raise RuntimeError(f"`x` is to large ({tuple(x.shape)}) to be padded to {tuple(shape)}")

    to, le = tl = padding // 2
    bo, ri = padding - tl
    padding = (le, ri, to, bo)

    additional_dims = (None,) * (4 - len(x.shape))
    if isinstance(value, Tensor) and len(value.shape) > 0:
        value = value.view(*value.shape, *([1] * (len(x.shape) - len(value.shape))))
        return nnF.pad(x[additional_dims] - value, padding, mode=mode, value=0)[
            (0,) * len(additional_dims)].add_(value)
    else:
        return nnF.pad(x[additional_dims], padding, mode=mode, value=value)[
            (0,) * len(additional_dims)]


PadToShape = func_to_module_class(pad_to_shape)


@make_multiinput
def crop(x, location: tuple, shape: tuple):
    """Crops an image.

    Args:
        x (Tensor or tuple): Input image (or tuple of images), a CHW array.
        location (int): Top left point of the cropping area
        shape (int): size.

    Returns:
        An array containing the cropped image.
    """
    (y0, x0), (h, w) = location, shape
    return x[..., y0:y0 + h, x0:x0 + w]


Crop = func_to_module_class(crop)


def paste_crop(x, crop, location):
    (y0, x0), (h, w) = location, crop.shape[-2:]
    y = x.clone()
    y[..., y0:y0 + h, x0:x0 + w] = crop
    return y


@make_multiinput
def hflip(x: Tensor) -> Tensor:
    return x.flip(-1)  # CHW


HFlip = func_to_module_class(hflip, name="HFlip")


# Random

def _resolve_padding(input_padding: T.Union[Number, str, T.Sequence], shape: T.Sequence):
    if isinstance(input_padding, Number):
        result = (input_padding, input_padding)
    elif input_padding == 'half':
        result = tuple(a / 2 for a in shape)
    elif isinstance(input_padding, T.Sequence) and len(input_padding) == 2:
        result = input_padding
    else:
        raise ValueError("Invalid `input_padding` argument value.")
    return np.array(result)


def random_crop_args(x: T.Union[Tensor, T.Sequence], shape, overflow=0):
    input_shape, shape = np.array(x[0].shape[-2:]), np.array(shape)
    overflow = _resolve_padding(overflow, shape)
    p0 = np.random.rand(2) * (input_shape - shape) - overflow / 2
    p1 = p0 + shape
    p0 = np.maximum(p0, 0, out=p0)  # in-place
    p1 = np.minimum(p1, input_shape, out=p1)

    feasible_shape = num.round_to_int(p1 - p0)
    return dict(location=num.round_to_int(p0), shape=feasible_shape)


def random_crop(x: T.Union[Tensor, T.Sequence], shape, overflow=0):
    return crop(x, **random_crop_args(x, shape, overflow=overflow))


RandomCrop = func_to_module_class(random_crop)


def random_hflip(x: Tensor, p=0.5) -> Tensor:
    return hflip(x) if np.random.rand() < p else x


RandomHFlip = func_to_module_class(random_hflip)


def random_scale_crop(x, shape, max_scale, min_scale=None, overstepping=0, is_segmentation=False):
    multiple = isinstance(x, tuple)
    xs = x if multiple else (x,)
    if isinstance(is_segmentation, bool):
        is_segmentation = [is_segmentation] * len(xs)
    if min_scale is None:
        min_scale = 1 / max_scale
    input_shape = xs[0].shape[-2:]
    if not all(a.shape[-2:] == input_shape for a in xs):
        raise RuntimeError("All inputs must have the same height and width.")

    scale = np.random.rand() * (max_scale - min_scale) + min_scale

    xs = random_crop(xs, shape=np.array(shape) / scale, overflow=overstepping)

    shape = tuple(
        d for d in
        np.minimum(np.array(shape), (np.array(xs[0].shape[-2:]) * scale + 0.5).astype(int)))
    xs = tuple(
        (resize_segmentation if iss else partial(resize, mode='bilinear'))(x, shape=shape)
        for x, iss in zip(xs, is_segmentation))
    return xs if multiple else xs[0]


RandomScaleCrop = func_to_module_class(random_scale_crop)
