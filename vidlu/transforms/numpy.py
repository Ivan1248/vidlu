import numpy as np


def center_crop(img, shape, fill=0, padding_mode='constant'):
    shape = np.array(shape)
    if len(shape) != 2 or np.any(shape <= 0):
        raise ValueError("shape should have length 2 and have positive elements.")
    dims_delta = (shape - img.shape[:2])
    negative = dims_delta < 0
    if np.any(negative):
        crop = -dims_delta * negative
        ca = np.round(crop / 2).astype(int)
        cb = crop - ca
        img = img[ca[0]:(-cb[0] or img.shape[0]), ca[1]:(-cb[1] or img.shape[1])]
    positive = dims_delta > 0
    if np.any(positive):
        padding = dims_delta * positive
        pa = np.round(padding / 2).astype(int)
        pb = padding - pa
        pad_width = [(pa[0], pb[0]), (pa[1], pb[1])] + [(0, 0) for _ in range(len(img.shape) - 2)]
        img = np.pad(img, pad_width, mode=padding_mode,
                     constant_values=fill)
    return img


def pad(image, padding: int):
    pad_width = [[padding] * 2] * 2
    if len(image.shape) == 3:
        pad_width.append([0, 0])
    return np.pad(image, pad_width, mode='constant')


def remap_segmentation(lab: np.ndarray, id_to_label):
    if len(lab.shape) == 3:  # for rgb labels
        scalarizer = np.array([256 ** 2, 256, 1])
        u, inv = np.unique(lab.reshape(-1, 3).dot(scalarizer), return_inverse=True)
        id_to_label = {np.array(k).dot(scalarizer): v for k, v in id_to_label.items()}
        return np.array([id_to_label.get(k, -1) for k in u], dtype=lab.dtype)[inv].reshape(
            lab.shape[:2])
    elif len(id_to_label) > 140:  # faster for great numbers of distinct labels
        u, inv = np.unique(lab, return_inverse=True)
        return np.array([id_to_label.get(k, k) for k in u], dtype=lab.dtype)[inv].reshape(lab.shape)
    else:  # faster for small numbers of distinct labels
        for id_, lb in id_to_label.items():
            lab[lab == id_] = lb
        return lab
