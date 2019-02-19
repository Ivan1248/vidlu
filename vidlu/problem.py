from enum import Enum


class Problem(Enum):
    CLASSIFICATION = 'classification'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'
    DEPTH_REGRESSION = 'depth_regression'
    OTHER = 'other'
