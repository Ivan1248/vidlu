from dataclasses import dataclass
import random
from functools import partial
from numbers import Number
from typing import Union, Sequence

import cv2
from torch import Tensor
import torch.nn.functional as nnF
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtF

from vidlu.utils import num
from vidlu.utils.func import func_to_class, class_to_func, make_multiinput
from vidlu.utils.torch import is_int_tensor, round_float_to_int
from . import numpy as npt


# Maybe have 1 class per transform, like in torchvision, instead of TRANSFORMERS
# Type and format checking #########################################################################

def is_pil_image(img):
    return isinstance(img, Image.Image)


def is_torch_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        return (f"{self.__class__.__name__}(\n    "
                + "\n    ".join(repr(t) for t in self.transforms) + "\n)")


# Conversions ######################################################################################

def pil_to_torch(img):
    """Converts a PIL Image to a Torch tensor with HWC (not CHW) layout with
    values in from 0 to 255.

    Taken from `torchvision.transforms` and modified.

    Args:
        img (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: HWC image with values from 0 to 255.
    """
    if not is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}.')

    if img.mode == 'I':  # i32, greyscale
        return torch.from_numpy(np.array(img, np.int32, copy=False))
    elif img.mode == 'I;16':  # i16, greyscale
        return torch.from_numpy(np.array(img, np.int16, copy=False))
    elif img.mode == '1':  # binary, stored with one pixel per byte
        return 255 * torch.from_numpy(np.array(img, np.uint8, copy=False))
    elif img.mode == 'F':
        raise ValueError("Float PIL-images not supported.")
    else:
        timg = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        # PIL image mode: L, P, RGB, YCbCr, RGBA, CMYK
        if img.mode == 'YCbCr':  # 24-bit color, video format
            nchannel = 3
        else:
            nchannel = len(img.mode)
        return timg.view(img.size[1], img.size[0], nchannel)


class PILToTorch:
    __call__ = staticmethod(pil_to_torch)


numpy_to_torch = torch.from_numpy


class NumPyToTorch:
    __call__ = staticmethod(torch.from_numpy)


def to_torch(x):
    if is_numpy_image(x):
        return numpy_to_torch(x)
    elif is_pil_image(x):
        return pil_to_torch(x)
    else:
        raise TypeError()


class ToTorch:
    __call__ = staticmethod(to_torch)


def torch_to_numpy(arr):
    return arr.numpy()


class TorchToNumPy:
    __call__ = staticmethod(torch_to_numpy)


def numpy_to_pil(npimg, mode=None):
    """Converts a NumPy array to a PIL Image.

    Taken from `torchvision.transforms` and modified.

    Args:
        npimg (torch.Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not is_numpy_image(npimg):
        raise TypeError(f'img should be an ndarray image. Got {type(npimg)}.')

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}."
                             + f" Should be {expected_mode}")
        mode = expected_mode
    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError(f'Input type {npimg.dtype} is not supported.')

    return Image.fromarray(npimg, mode=mode)


@dataclass
class NumPyToPIL:
    mode: str = None

    def __call__(self, img):
        return numpy_to_pil(img, mode=self.mode)


def pil_to_numpy(img, dtype=None, copy=False):
    return np.array(img, dtype=dtype, copy=copy)


class PILToNumPy:
    __call__ = staticmethod(np.array)  # keywords: call, copy, ...


def torch_to_pil(timg, mode=None):
    """Converts a Torch tensor to a PIL Image.

    Taken from `torchvision.transforms` and modified.

    Args:
        timg (torch.Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not is_torch_image(timg):
        raise TypeError(f'x should be torch.Tensor image. Got {type(timg)}.')
    return numpy_to_pil(timg.numpy(), mode=mode)


@dataclass
class TorchToPIL:
    mode: str = None

    def __call__(self, img):
        return torch_to_pil(img, mode=self.mode)


def to_pil(x, mode=None):
    if is_torch_image(x):
        return torch_to_pil(x, mode=mode)
    elif is_numpy_image(x):
        return numpy_to_pil(x, mode=mode)
    else:
        raise ValueError("The input is neither a Torch nor a NumPy array.")


# NumPy ############################################################################################

class NumPyCenterCrop:
    def __init__(self, output_size, fill=0, padding_mode='constant'):
        self.kwargs = {k: v for k, v in locals().items() if
                       k in ['output_size', 'fill', 'padding_mode']}

    def __call__(self, x):
        return npt.center_crop(x, **self.kwargs)


class NumPyHFlip:
    def __call__(self, x):
        return np.fliplr(x)


def numpy_segmentation_edge_distance_transform(segmentation, class_count=None):
    present_classes = np.unique(segmentation)
    if class_count is None:
        class_count = present_classes[-1]
    distances = np.full([class_count] + list(segmentation.shape), -1, dtype=np.float32)
    for i in present_classes if present_classes[0] >= 0 else present_classes[1:]:
        class_mask = segmentation == i
        distances[i][class_mask] = cv2.distanceTransform(
            np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
    return distances


# PIL ##############################################################################################

class PILHFlip:
    def __call__(self, x):
        return tvtF.hflip(x)


PILCenterCrop = tvt.CenterCrop
PILPad = tvt.Pad

PILRandomCrop = tvt.RandomCrop
PILRandHFlip = tvt.RandomHorizontalFlip

PILResize = tvt.Resize


class PILRescale:
    def __init__(self, factor, interpolation=2):
        self.resize = PILResize(size=(np.array(self.x.shape[:2]) * factor).astype(np.int),
                                interpolation=interpolation)

    def __call__(self, x):
        return self.resize(x)


# Torch ############################################################################################
# layout: CHW

@make_multiinput
def hwc_to_chw(x):
    return x.permute(2, 0, 1)


class HWCToCHW:
    __call__ = staticmethod(hwc_to_chw)  # keywords: call, copy, ...


@make_multiinput
def chw_to_hwc(x):
    return x.permute(1, 2, 0)


class CHWToHWC:
    __call__ = staticmethod(chw_to_hwc)  # keywords: call, copy, ...


class To:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, x):
        return x.to(*self.args, **self.kwargs)


@make_multiinput
def mul(x, factor):
    return x * factor


Mul = func_to_class(mul)


@make_multiinput
def div(x, divisor):
    return x / divisor


Div = func_to_class(div)


class Standardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return (x - self.mean.to(**fmt)) / self.std.to(**fmt)


standardize = class_to_func(Standardize)


class Destandardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        fmt = dict(dtype=x.dtype, device=x.device)
        return x * self.std.to(**fmt) + self.mean.to(**fmt)


destandardize = class_to_func(Destandardize)


@make_multiinput
def resize(x, shape=None, scale_factor=None, mode='nearest', align_corners=False):
    additional_dims = [None] * (4 - len(x.shape))
    return nnF.interpolate(x[additional_dims], size=shape, scale_factor=scale_factor, mode=mode,
                           align_corners=align_corners)[(0,) * len(additional_dims)]


Resize = func_to_class(resize)


@make_multiinput
def pad(x, padding, mode='constant', value=0):
    if isinstance(padding, int):
        padding = (padding,) * 4
    additional_dims = [None] * (4 - len(x.shape))
    return nnF.pad(x[additional_dims], padding, mode=mode, value=value)[(0,) * len(additional_dims)]


Pad = func_to_class(pad)


@make_multiinput
def pad_to_shape(x, shape, mode='constant', value=0):
    padding = np.array(shape) - np.array(x.shape[-2:])
    if np.all(padding == 0):
        return x
    elif np.any(padding < 0):
        raise RuntimeError(f"`x` is to large ({tuple(x.shape)}) to be padded to {tuple(shape)}")

    t, l = tl = padding // 2
    b, r = padding - tl
    padding = (l, r, t, b)

    additional_dims = [None] * (4 - len(x.shape))
    if isinstance(value, Tensor) and len(value.shape) > 0:
        value = value.view(*value.shape, *([1] * (len(x.shape) - len(value.shape))))
        return nnF.pad(x[additional_dims] - value, padding, mode=mode, value=0)[
            (0,) * len(additional_dims)].add_(value)
    else:
        return nnF.pad(x[additional_dims], padding, mode=mode, value=value)[
            (0,) * len(additional_dims)]


PadToShape = func_to_class(pad_to_shape)


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
    y0, x0 = location
    h, w = shape
    orig_shape = x.shape
    r = x.view(-1, *orig_shape[-2:])[:, y0:y0 + h, x0:x0 + w]
    return r.view(orig_shape[:-2] + r.shape[-2:])


Crop = func_to_class(crop)


@make_multiinput
def hflip(x: Tensor) -> Tensor:
    return x.flip(-1)  # CHW


HFlip = func_to_class(hflip, name="HFlip")


def random_crop(x: Union[Tensor, Sequence], shape, overstepping=0):
    input_shape, shape = np.array(x[0].shape[-2:]), np.array(shape)
    overstepping = _resolve_padding(overstepping, shape)
    p0 = np.random.rand(2) * (input_shape - shape) - overstepping / 2
    p1 = p0 + shape
    p0 = np.maximum(p0, 0, out=p0)  # in-place
    p1 = np.minimum(p1, input_shape, out=p1)

    feasible_shape = num.round_to_int(p1 - p0)
    return crop(x, location=num.round_to_int(p0), shape=feasible_shape)


RandomCrop = func_to_class(random_crop)


def random_hflip(x: Tensor, p=0.5) -> Tensor:
    return hflip(x) if random.random() < p else x


RandomHFlip = func_to_class(random_hflip)


def _resolve_padding(input_padding: Union[Number, str, Sequence], shape: Sequence):
    if isinstance(input_padding, Number):
        result = (input_padding, input_padding)
    elif input_padding == 'half':
        result = tuple(a / 2 for a in shape)
    elif isinstance(input_padding, Sequence) and len(input_padding) == 2:
        result = input_padding
    else:
        raise ValueError("Invalid `input_padding` argument value.")
    return np.array(result)


def resize_segmentation(x, shape=None, scale_factor=None, align_corners=None):
    if not is_int_tensor(x):
        raise TypeError("`x` should be an integer tensor.")
    xfr = resize(x.float(), shape=shape, scale_factor=scale_factor, mode='nearest',
               align_corners=align_corners)
    return round_float_to_int(xfr, x.dtype)


ResizeSegmentation = func_to_class(resize_segmentation)


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

    xs = random_crop(xs, shape=np.array(shape) / scale, overstepping=overstepping)

    shape = tuple(
        d for d in np.minimum(np.array(shape), (np.array(xs[0].shape[-2:]) * scale+0.5).astype(int)))
    xs = tuple(
        (resize_segmentation if iss else partial(resize, mode='bilinear'))(x, shape=shape)
        for x, iss in zip(xs, is_segmentation))
    return xs if multiple else xs[0]


RandomScaleCrop = func_to_class(random_scale_crop)

_this = (lambda: None).__module__
__all__ = [k for k, v in locals().items()
           if hasattr(v, '__module__') and v.__module__ == _this and not k[0] == '_']
