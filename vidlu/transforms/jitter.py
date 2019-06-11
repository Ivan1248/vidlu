import random

from .transformers import image_transformer


def cifar_jitter(x):
    pilt = image_transformer(x).to_pil()
    return pilt.rand_crop(pilt.item.size, padding=4).rand_hflip().item


def rand_hflip(*arrays, p=0.5):
    """
    Args:
        *arrays: fields of a single example, e.g. image and its segmentation.
        p (float): flip probability.

    Returns:
        An array or a tuple of arrays.
    """
    if p < random.random:
        arrays = [image_transformer(a).hflip() for a in arrays]
    if len(arrays) == 1:
        return arrays[0]
    return arrays[0] if len(arrays) == 1 else arrays
