from dataclasses import dataclass

from torch import Tensor, nn
from PIL import Image
import numpy as np
import torch


def is_pil_image(img):
    return isinstance(img, Image.Image)


def is_torch_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Type conversions #################################################################################

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


def pil_to_numpy(img, dtype=None, copy=False):
    return np.array(img, dtype=dtype, copy=copy)


class PILToNumPy:
    __call__ = staticmethod(np.array)  # keywords: call, copy, ...


def to_numpy(x):
    if is_pil_image(x):
        return pil_to_numpy(x)
    elif is_torch_image(x):
        return torch_to_numpy(x)
    else:
        raise TypeError()


class ToNumpy:
    __call__ = staticmethod(to_numpy)


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


class ToPIL:
    __call__ = staticmethod(to_pil)
