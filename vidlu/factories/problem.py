from dataclasses import dataclass
from enum import Enum
import typing as T

class Problem(Enum):
    CLASSIFICATION = 'classification'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'
    DEPTH_REGRESSION = 'depth_regression'
    OTHER = 'other'


@dataclass
class ProblemInfo:
    pass


@dataclass
class Supervised(ProblemInfo):
    class_count: int


@dataclass
class Classification(Supervised):
    class_count: int


@dataclass
class SemanticSegmentation(Classification):
    y_shape: T.Tuple[int, int]


@dataclass
class DepthRegression(Supervised):
    y_shape: T.Tuple[int, int]


_name_to_class = dict(classification=Classification, semantic_segmentation=SemanticSegmentation)


def get_problem_type(name: str):
    return _name_to_class[name]


def get_problem(name: str, *args, **kwargs):
    return get_problem_type(name)(*args, **kwargs)
