import numpy as np


def center_crop(img, shape, padding_value=0):
    old_dims = img.shape[:2]
    new_dims = np.array(shape, dtype=np.int16)[:2]
    dims_delta = (new_dims - old_dims)
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


def crop(image: Image.Image, size, error_on_too_small=True):
    d = [max(0, image.shape[i] - maxshape[i]) for i in im.size]
    if (np.sum(d) == 0):
        return image
    return image[(d[0] + 1) // 2:image.shape[0] - (d[0] // 2), (d[1] + 1) // 2:
                                                               image.shape[1] - (d[1] // 2)]


def pad(image, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant')


def pad_to_shape(image, shape, value=1):
    d = [shape[i] - image.shape[i] for i in [0, 1]]
    pad_width = [((d[0] + 1) // 2, d[0] // 2), ((d[1] + 1) // 2, d[1] // 2)]
    if len(image.shape) == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode='constant', constant_values=value)


def center_to_shape(image, shape):
    return pad_to_shape(crop(image, shape), shape)


def fill_to_shape(image, shape):
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image
    ratio = max(shape[0] / image.shape[0], shape[1] / image.shape[1])
    shape_non_cropped = [round(ratio * x) for x in image.shape[:2]]
    image = imresize(image, shape, interp='bilinear')
    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image
    return crop(image, shape)
