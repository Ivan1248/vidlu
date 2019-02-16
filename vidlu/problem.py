from enum import Enum

import numpy as np


class Problem(Enum):
    CLASSIFICATION = 'classification'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'
    DEPTH_REGRESSION = 'depth_regression'
    OTHER = 'other'


def dataset_to_problem(dataset):
    if 'problem' not in dataset.info:
        raise ValueError("Unknown problem.")
    return Problem(dataset.info.problem)
    """
    if len(example) == 2:
        x, y = example
        if 'class_count' in info:
            if isinstance(y, (int, np.integer)):
                return Problem.CLASSIFICATION
            elif x.shape[1:] == y.shape and len(y.shape) == 2:
                return Problem.SEMANTIC_SEGMENTATION
            else:
                raise ValueError("Unknown classification problem.")
        elif x.shape[1:] == y.shape and len(y.shape) == 2 and isinstance(y[0, 0], float):
            return Problem.DEPTH_REGRESSION
        else:
            raise ValueError("Unknown problem.")
    elif len(example) == 1:
        return Problem.OTHER
    """
