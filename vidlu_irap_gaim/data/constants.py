import cv2
import numpy as np


RGB_MEAN: tuple[float, float, float] = (0.53354913, 0.52727484, 0.48752149)
RGB_STD: tuple[float, float, float] = (0.20401913, 0.20417478, 0.25402164)
INPUT_DIM_RGB: tuple[int, int, int] = (384, 288, 3)


def _load_image_cv2(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
