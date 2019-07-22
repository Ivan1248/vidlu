from dataclasses import dataclass
from functools import partial

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


# Conversion #######################################################################################


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
        #elif npimg.dtype == np.float32:
        #    expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}." +
                             f" Should be {expected_mode}")
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


# Transformers #####################################################################################


class Transformer:
    __slots__ = 'item'

    def __init__(self, item, **args):
        self.item = item

    def transform(self, func, *args, **kwargs):
        return type(self)(**{**self.__dict__, 'item': func(self.item, *args, **kwargs)})


class ImageTransformer(Transformer):
    __slots__ = 'layout'

    def __init__(self, item, layout='HWC'):
        super().__init__(item)
        self.layout = layout

    def transform(self, func, *args, **kwargs):
        if isinstance(func, str):
            return getattr(type(self))
        return super().transform(func, *args, **kwargs)


class NumPyImageTransformer(ImageTransformer):

    def __init__(self, item, layout='HWC'):
        if layout != 'HWC':
            raise ValueError('The memory layout of the array should be HWC.')
        super().__init__(item, layout)

    def to_numpy(self, dtype=None):
        return (self if dtype is None or dtype == self.item.dtype else np.array(
            self.item, dtype=dtype))

    def to_pil(self, mode=None):
        return PILImageTransformer(numpy_to_pil(self.item, mode=mode))

    def to_torch(self):
        return TorchImageTransformer(numpy_to_torch(self.item))

    def center_crop(self, output_size, fill=0, padding_mode='constant'):
        return self.transform(npt.center_crop, output_size, fill=fill, padding_mode=padding_mode)

    def hflip(self):
        return self.transform(np.fliplr)


class PILImageTransformer(ImageTransformer):

    def to_numpy(self, dtype=None, copy=False):
        return NumPyImageTransformer(pil_to_numpy(self.item, dtype=dtype, copy=copy))

    def to_pil(self, mode=None):
        return self if mode is None else PILImageTransformer(self.item.convert(mode))

    def to_torch(self):
        return TorchImageTransformer(pil_to_torch(self.item))

    def center_crop(self, output_size):
        return self.transform(F.center_crop, output_size)

    def pad(self, padding, fill=0, padding_mode='constant'):
        return self.transform(F.pad, padding, fill=fill, padding_mode=padding_mode)

    def hflip(self):
        return self.transform(F.hflip)

    def rand_crop(self, size, padding=0, pad_if_needed=False):
        return self.transform(tvt.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed))

    def rand_hflip(self, p=0.5):
        return self.transform(tvt.RandomHorizontalFlip(p))


class TorchImageTransformer(ImageTransformer):

    def to_numpy(self):
        return NumPyImageTransformer(torch_to_numpy(self.item), layout=self.layout)

    def to_pil(self, mode=None):
        if self.layout != 'HWC':
            raise ValueError("Cannot convert array with layout CHW to PIL." +
                             " Use chw_to_hwc first.")
        return PILImageTransformer(torch_to_pil(self.item, mode=mode))

    def to_torch(self, dtype=None):
        return (self if dtype is None or dtype == self.item.dtype else torch.tensor(
            self.item, dtype=dtype))

    def hwc_to_chw(self):
        if self.layout == 'HWC':
            return TorchImageTransformer(self.item.permute(2, 0, 1), layout='CHW')
        raise ValueError("Only HWC to CHW transposition is possible.")

    def chw_to_hwc(self):
        if self.layout == 'CHW':
            return TorchImageTransformer(self.item.permute(1, 2, 0), layout='HWC')
        raise ValueError("Only CHW to HWC transposition is possible.")

    def to_float32(self):
        return self.transform(lambda x: x.float())

    def to_uint8(self):
        return self.transform(lambda x: x.byte())

    def div255(self):
        return self.transform(lambda x: x / 255)

    def mul255(self):
        return self.transform(lambda x: x * 255)

    def standardize(self, mean, std, dtype=torch.float32):
        mean, std = [torch.tensor(s, dtype=dtype) for s in [mean, std]]
        return self.transform(lambda x: (x.to(dtype) - mean) / std)

    def destandardize(self, mean, std):
        if not isinstance(self.item, torch.FloatTensor):
            raise TypeError("Only float tensors can be destandardized.")
        mean, std = [torch.tensor(s, dtype=torch.float32) for s in [mean, std]]
        return self.transform(lambda x: x * std + mean)

    def resize(self, size, interpolation=2):
        self.transform(partial(tvt.functional.resize, size=size, interpolation=interpolation))

    def scale(self, factor, interpolation=2):
        self.resize(
            (np.array(self.x.shape[:2]) * factor).astype(np.int), interpolation=interpolation)


def image_transformer(x):
    if is_numpy_image(x):
        return NumPyImageTransformer(x)
    elif is_pil_image(x):
        return PILImageTransformer(x)
    elif is_torch_image(x):
        return TorchImageTransformer(x)
    else:
        raise TypeError("Unsupported type.")
