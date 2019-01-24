import skimage.util, skimage.transform
import numpy as np
from scipy.misc import imresize


def crop(image: np.ndarray, maxshape):
    d = [max(0, image.shape[i] - maxshape[i]) for i in [0, 1]]
    if (np.sum(d) == 0):
        return image
    return image[(d[0] + 1) // 2:image.shape[0] - (d[0] // 2), (d[1] + 1) // 2:
                 image.shape[1] - (d[1] // 2)]


def pad(image: np.ndarray, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')


def pad_to_shape(image: np.ndarray, shape, value=1):
    d = [shape[i] - image.shape[i] for i in [0, 1]]
    pad_width = [((d[0] + 1) // 2, d[0] // 2), ((d[1] + 1) // 2, d[1] // 2)]
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant', constant_values=value)


def center_to_shape(image: np.ndarray, shape):
    return pad_to_shape(crop(image, shape), shape)


def fill_to_shape(image: np.ndarray, shape):
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image
    ratio = max(shape[0] / image.shape[0], shape[1] / image.shape[1])
    shape_non_cropped = [round(ratio * x) for x in image.shape[:2]]
    image = imresize(image, shape, interp='bilinear')
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image
    return crop(image, shape)
