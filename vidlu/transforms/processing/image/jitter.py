import numpy as np
from vidlu.transforms.numpy import pad


def random_crop(im, shape):
    d = [im.shape[i] - shape[i] + 1 for i in [0, 1]]
    d = list(map(np.random.randint, d))
    return im[d[0]:d[0] + shape[0], d[1]:d[1] + shape[1]]


def random_crop_with_label(im_lab, shape):
    im, lab = im_lab
    d = [im.shape[i] - shape[i] + 1 for i in [0, 1]]
    d = list(map(np.random.randint, d))
    return (im[d[0]:d[0] + shape[0], d[1]:d[1] + shape[1]],
            lab[d[0]:d[0] + shape[0], d[1]:d[1] + shape[1]])


def random_fliplr(im):
    return np.fliplr(im) if np.random.rand() < .5 else im


def random_fliplr_with_label(im_lab):
    return tuple(map(np.fliplr, im_lab)) if np.random.rand() < .5 else im_lab


def cifar_jitter(im, max_padding=4):
    shape = im.shape[:2]
    im = pad(im, max_padding)
    im = random_crop(im, shape)
    return random_fliplr(im)
