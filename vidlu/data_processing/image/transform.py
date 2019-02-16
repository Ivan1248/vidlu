import cv2
import numpy as np


def translate(image: np.ndarray, delta: tuple) -> np.ndarray:  # TODO: test
    rows, cols = image.shape
    M = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)


def zoom(image: np.ndarray, proportion: float) -> np.ndarray:  # TODO: test
    rows, cols = image.shape
    M = np.float32([[proportion, 0, 0], [0, proportion, 0]])
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:  # TODO: test
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)


def transform(image: np.ndarray, rotation_deg: float = 0, scale: float = 1, flip: bool = False):
    rows, cols = image.shape[0], image.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_deg, scale)
    if flip:
        for i in range(2):
            M[i, 2] += cols * M[i, 0]
            M[i, 0] *= -1
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = cv2.imread('test.png')
    b = np.load('test.label.npy')
    a1 = transform(a, 4, 1.09, True)
    b1 = transform(b, 4, 1.09, True)
    cv2.imshow('a', a1)
    cv2.waitKey(4000)
    cv2.imshow('a', b1)
    cv2.waitKey(4000)
