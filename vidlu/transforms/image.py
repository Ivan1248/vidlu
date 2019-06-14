from dataclasses import dataclass
from functools import partial
from typing import Any

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
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
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


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


class ToTorch:
    def __call__(self, x):
        if is_numpy_image(x):
            return numpy_to_torch(x)
        elif is_pil_image(x):
            return pil_to_torch(x)
        else:
            raise TypeError()


class PILToTorch:
    __call__ = staticmethod(pil_to_torch)


numpy_to_torch = torch.from_numpy


class NumPyToTorch:
    __call__ = staticmethod(torch.from_numpy)


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


class NumPyCenterCrop:
    def __init__(self, output_size, fill=0, padding_mode='constant'):
        self.kwargs = {k: v for k, v in locals().items() if
                       k in ['output_size', 'fill', 'padding_mode']}

    def __call__(self, x):
        return npt.center_crop(x, **self.kwargs)


class NumPyHFlip:
    def __call__(self, x):
        return np.fliplr(x)


class PILHFlip:
    def __call__(self, x):
        return F.hflip(x)


PILCenterCrop = tvt.CenterCrop
PILPad = tvt.Pad

PILRandomCrop = tvt.RandomCrop
PILRandHFlip = tvt.RandomHorizontalFlip


class HWCToCHW:
    def __call__(self, x):
        return x.permute(2, 0, 1)


class CHWToHWC:
    def __call__(self, x):
        return x.permute(1, 2, 0)


class To:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, x):
        return x.to(*self.args, **self.kwargs)


@dataclass
class Mul:
    factor: Any

    def __call__(self, x):
        return x * self.factor


@dataclass
class Div:
    divisor: Any

    def __call__(self, x):
        return x / self.divisor


class Standardize:
    def __init__(self, mean, std):
        self.mean, self.std = mean.view(-1, 1, 1), std.view(-1, 1, 1)

    def __call__(self, x):
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)


class Destandardize:
    def __init__(self, mean, std, dtype):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return x * self.std + self.mean


Resize = tvt.Resize


class Scale:
    def __init__(self, factor, interpolation=2):
        self.resize = Resize(size=(np.array(self.x.shape[:2]) * factor).astype(np.int),
                             interpolation=interpolation)

    def __call__(self, x):
        return self.resize(x)
