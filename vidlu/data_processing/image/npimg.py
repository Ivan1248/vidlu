import numpy as np


def center_crop(img, shape, padding_value=0):
    dims_delta = (img.shape[:2] - np.array(shape, dtype=np.int16)[:2])
    negative = dims_delta < 0
    if np.any(negative):
        crop = -dims_delta * negative
        ca = np.round(crop / 2)
        cb = crop - ca
        img = img[ca[0]:-cb[0], ca[1]:-cb[1]]
    positive = dims_delta > 0
    if np.any(positive):
        padding = dims_delta * positive
        pa = np.round(padding / 2)
        pb = padding - pa
        img = np.pad(img, [[pa[0], pb[0]], [pa[1], pb[1]]], mode='constant',
                     constant_values=padding_value)
    return img


def pad(image, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')


"""
def pad(img, shape, padding_value=0):
    dims_delta = (img.shape[:2] - np.array(shape, dtype=np.int16)[:2])
    positive = dims_delta > 0
    negative = dims_delta < 0
    if anp.any(negative):
        raise ValueError("")
    if np.any(positive):
        padding = dims_delta * positive
        pa = np.round(padding / 2)
        pb = padding - pa
        img = np.pad(img, [[pa[0], pb[0]], [pa[1], pb[1]]], mode='constant',
                     constant_values=padding_value)
    return img


def crop(img, shape, padding_value=0):
    dims_delta = (img.shape[:2] - np.array(shape, dtype=np.int16)[:2])
    negative = dims_delta < 0
    if np.any(negative):
        crop = -dims_delta * negative
        ca = np.round(crop / 2)
        cb = crop - ca
        img = img[ca[0]:-cb[0], ca[1]:-cb[1]]
    return img
"""
