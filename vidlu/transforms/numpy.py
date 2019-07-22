import numpy as np


def center_crop(img, shape, fill=0, padding_mode='constant'):
    shape = np.array(shape)
    if len(shape) != 2 or np.any(shape <= 0):
        raise ValueError("shape should have length 2 and have positive elements.")
    dims_delta = (shape - img.shape[:2])
    negative = dims_delta < 0
    if np.any(negative):
        crop = -dims_delta * negative
        ca = np.round(crop / 2).astype(np.int)
        cb = crop - ca
        img = img[ca[0]:(-cb[0] or img.shape[0]), ca[1]:(-cb[1] or img.shape[1])]
    positive = dims_delta > 0
    if np.any(positive):
        padding = dims_delta * positive
        pa = np.round(padding / 2).astype(np.int)
        pb = padding - pa
        pad_width = [(pa[0], pb[0]), (pa[1], pb[1])] + [(0, 0) for _ in range(len(img.shape) - 2)]
        img = np.pad(img, pad_width, mode=padding_mode,
                     constant_values=fill)
    return img


def pad(image, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')
