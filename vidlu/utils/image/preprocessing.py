import numpy as np


def get_normalization_statistics(images):
    return images.mean((0, 1, 2)), images.std((0, 1, 2))


def denormalize(images, mean, std):
    return images * std + mean
