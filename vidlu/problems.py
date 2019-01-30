from enum import Enum


class Problems(Enum):
    CLASSIFICATION = 'clf'
    SEMANTIC_SEGMENTATION = 'semseg'
    DEPTH_REGRESSION = 'dregr'
    OTHER = 'oth'


def dataset_to_problem(dataset):
    info = dataset.info
    example = dataset[0]
    if len(example) == 2:
        x, y = example
        if 'class_count' in info:
            if isinstance(dataset[0][1], int):
                return Problems.CLASSIFICATION
            elif x.shape[1:] == y.shape and len(y.shape) == 2:
                return Problems.SEMANTIC_SEGMENTATION
            else:
                raise ValueError("Unknown classification problem.")
        elif x.shape[1:] == y.shape and len(y.shape) == 2 and isinstance(y[0, 0], float):
            return Problems.DEPTH_REGRESSION
        else:
            raise ValueError("Unknown problem")
    elif len(example) == 1:
        return Problems.OTHER
