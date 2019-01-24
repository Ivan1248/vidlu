import numpy as np


def get_normalization_statistics(images):
    return images.mean((0, 1, 2)), images.std((0, 1, 2))


def normalize(images, mean=None, std=None):
    assert (mean is None) == (std is None)
    if mean is None:
        mean, std = get_normalization_statistics(images)
    return (images - mean) / std


def denormalize(images, mean, std):
    return images * std + mean
