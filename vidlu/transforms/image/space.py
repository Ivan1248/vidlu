from numbers import Number
import typing as T

import torch
from torch import Tensor
import torch.nn.functional as nnF
import numpy as np
from typeguard import check_argument_types

import vidlu.data.types as dt
from vidlu.utils import num
from vidlu.utils.func import partial
from vidlu.utils.func import vectorize
from vidlu.torch_utils import is_int_tensor, round_float_to_int
from vidlu.modules.utils import func_to_module_class


@vectorize
def resize(x, shape=None, scale_factor=None, mode='nearest', align_corners=None):
    additional_dims = (None,) * (4 - len(x.shape))  # must be tuple, empty list doesn't work
    if isinstance(x, torch.Tensor):
        return nnF.interpolate(x[additional_dims], size=shape, scale_factor=scale_factor, mode=mode,
                               align_corners=align_corners)[(0,) * len(additional_dims)]
    scale = shape / x.shape
    if isinstance(x, dt.AABBsOnImageCollection):
        return x.map(lambda bb: bb * scale, shape=shape)
    elif isinstance(x, dt.AABB):
        return x * scale
    raise TypeError(f"Unsupported type {type(x)}.")


Resize = func_to_module_class(resize)


def resize_segmentation(x, shape=None, scale_factor=None, align_corners=None):
    if not is_int_tensor(x):
        raise TypeError("`x` should be an integer tensor.")
    xfr = resize(x.float(), shape=shape, scale_factor=scale_factor, mode='nearest',
                 align_corners=align_corners)
    return round_float_to_int(xfr, x.dtype)


ResizeSegmentation = func_to_module_class(resize_segmentation)


@vectorize
def pad(x, padding, mode='constant', value=0):
    if isinstance(padding, int):
        padding = (padding,) * 4  # left, right, top, bottom
    (le, ri, to, bo) = padding

    if isinstance(x, torch.Tensor):
        additional_dims = (None,) * (4 - len(x.shape))
        if isinstance(x, dt.Image):
            if value == 'mean':
                value = x.mean((1, 2))
            if hasattr(value, 'shape') and len(value.shape) > 0:
                value = value.view(*value.shape, *([1] * (len(x.shape) - len(value.shape))))
                return nnF.pad(x[additional_dims] - value, padding, mode=mode, value=0)[
                    (0,) * len(additional_dims)].add_(value)
        return nnF.pad(x[additional_dims], padding, mode=mode, value=value)[
            (0,) * len(additional_dims)]
    elif isinstance(x, dt.AABB):
        return x + np.array([le, to])
    elif isinstance(x, dt.AABBsOnImageCollection):
        return x.map(lambda bb: bb + np.array([le, to]),
                     shape=x.shape + np.array([le + ri, to + bo]))
    else:
        raise TypeError(f"Unsupported type {type(x).__name__}.")


Pad = func_to_module_class(pad)


@vectorize
def pad_to_shape(x, shape, mode='constant', value=0):
    if value == 'mean':
        value = x.mean((1, 2))
    padding = np.array(shape) - np.array(x.shape[-2:])
    if np.any(padding < 0):
        raise RuntimeError(f"`x` is to large ({tuple(x.shape)}) to be padded to {tuple(shape)}")

    if np.all(padding == 0):
        return x

    to, le = tl = padding // 2
    bo, ri = padding - tl
    padding = (le, ri, to, bo)

    return pad(x, padding, mode, value)


PadToShape = func_to_module_class(pad_to_shape)


def crop(x, aabb):
    """Crops an image.

    Args:
        x (Tensor or tuple): Input image (or tuple of images), a CHW array.
        location (int): Top left point of the cropping area
        shape (int): size.

    Returns:
        An array containing the cropped image.
    """
    if isinstance(x, torch.Tensor):
        return x[..., aabb.min[0]:aabb.max[0], aabb.min[1]:aabb.max[1]]
    elif isinstance(x, dt.AABB):
        return x.clip(max=aabb.max) - aabb.min
    elif isinstance(x, dt.AABBsOnImageCollection):
        return x.map(partial(crop, aabb), shape=aabb.size)
    else:
        raise TypeError(f"Unsupported type {type(x).__name__}.")


Crop = func_to_module_class(crop)


@vectorize
def hflip(x: Tensor) -> Tensor:
    if isinstance(x, dt.ClassAABBsOnImage):
        return x.map(lambda bb: type(bb)(min=[x.shape[0] - bb.min[0], bb.min[1]],
                                         max=[x.shape[0] - bb.max[0], bb.max[1]]))
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


def random_crop_box(image_shape, shape, overflow=0, rng=np.random):
    input_shape, shape = np.array(image_shape[-2:]), np.array(shape)
    overflow = _resolve_padding(overflow, shape)
    p0 = rng.rand(2) * (input_shape - shape) - overflow / 2
    return dt.AABB(min=p0, size=shape).clip(min=(0, 0), max=input_shape).map(num.round_to_int)


def random_overlapping_crop_box(image_shape, shape, aabbs, overflow=0,
                                min_overlap_proportion=1e-6, rng=np.random):
    areas = np.array([b.area for b in aabbs])
    if len(aabbs) == 0 or sum(areas) == 0:
        return random_crop_box(image_shape, shape, overflow=overflow, rng=rng)
    for _ in range(100):
        result = random_crop_box(image_shape, shape, overflow=overflow, rng=rng)
        overlaps = np.array([result.intersect(b).area for b in aabbs])
        if np.max(np.nan_to_num(overlaps / areas)) > min_overlap_proportion:
            break
    return result


@vectorize(n=2)
def rare_class_random_overlapping_crop_box(x: T.Union[Tensor, T.Sequence], class_to_aabbs, shape,
                                           rare_classes, overflow=0,
                                           min_overlap_proportion=1e-6, rng=np.random):
    aabbs = []
    for c in rare_classes:
        aabbs.extend(class_to_aabbs[c])
    return random_overlapping_crop_box(x, shape=shape, overflow=overflow, aabbs=aabbs,
                                       min_overlap_proportion=min_overlap_proportion, rng=rng)


def random_crop(x: T.Union[dt.Spatial2D, T.Sequence], shape, overflow=0, rng=np.random):
    if isinstance(x, tuple):
        image = next(a for a in x if isinstance(a, dt.ArraySpatial2D))
    else:
        image = x
    return vectorize(crop)(x, random_crop_box(image.shape, shape, overflow=overflow, rng=rng))


RandomCrop = func_to_module_class(random_crop)


def random_hflip(x: Tensor, p=0.5, rng=np.random) -> Tensor:
    return hflip(x) if rng.rand() < p else x


RandomHFlip = func_to_module_class(random_hflip)

ScaleDistArg = T.Literal["uniform", "inv-uniform", "log-uniform"]


def _sample_random_scale(min, max, dist: ScaleDistArg = "uniform", rng=np.random):
    if min is None:
        min = 1 / max
    if dist == "log-uniform":
        scale = np.exp(rng.uniform(np.log(min), np.log(max)))
    elif dist == "inv-uniform":
        scale = 1 / (rng.uniform(1 / max, 1 / min))
    else:
        scale = rng.uniform(min, max)
    return scale


def random_scale_crop(x, shape, max_scale, min_scale=None, overflow=0,
                      align_corners=None, scale_dist: ScaleDistArg = "uniform", rng=np.random,
                      rand_crop=random_crop):
    check_argument_types()

    multiple = isinstance(x, tuple)
    xs = x if multiple else (x,)

    input_shape = xs[0].shape[-2:]
    if not all(tuple(a.shape[-2:] if isinstance(a, torch.Tensor) else a.shape) == input_shape
               for a in xs if isinstance(a, dt.Spatial2D)):
        raise RuntimeError("All inputs must have the same height and width.")

    scale = _sample_random_scale(min_scale, max_scale, dist=scale_dist, rng=rng)

    xs = rand_crop(xs, shape=np.array(shape) / scale, overflow=overflow, rng=rng)

    shape = [d for d in
             np.minimum(np.array(shape), (np.array(xs[0].shape[-2:]) * scale + 0.5).astype(int))]
    xs = tuple(resize_segmentation(x, shape=shape) if isinstance(x, dt.SegMap) else
               resize(x, shape=shape, mode='bilinear', align_corners=align_corners)
               for x in xs)
    return xs if multiple else xs[0]


RandomScaleCrop = func_to_module_class(random_scale_crop)
