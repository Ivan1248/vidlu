from functools import partial
import dataclasses as dc
import typing as T
from enum import Enum

from vidlu import metrics


def get_universal_metrics():
    return [partial(metrics.MaxMultiMetric, filter=lambda k, v: k.startswith('mem')),
            # partial(metrics.HarmonicMeanMultiMetric, filter=lambda k, v: k.startswith('freq')),
            # partial(metrics.with_suffix(metrics.MaxMultiMetric, 'max'),
            #         filter=lambda k, v: k.startswith('freq')),
            partial(metrics.MedianMultiMetric, filter=lambda k, v: k.startswith('freq')),
            partial(metrics.AverageMultiMetric, filter=lambda k, v: k.startswith('loss'))]


def get_classification_metrics(problem, metric_names):
    result = get_universal_metrics()
    # common_names = ['mem', 'freq', 'loss', 'A', 'mIoU']
    # result.append(partial(metrics.AverageMultiMetric,
    #                       filter=lambda k, v: isinstance(v, (int, float)) and not (
    #                           any(k.startswith(c) for c in common_names))))
    get_hard_prediction = lambda r: r.out.argmax(1)
    if ProblemExtra.ADV in problem.extra:
        result.append(partial(metrics.with_suffix(metrics.ClassificationMetrics, 'adv'),
                              get_hard_prediction=get_hard_prediction,
                              class_count=problem.class_count, metrics=metric_names))
    result.append(
        partial(metrics.ClassificationMetrics, get_hard_prediction=get_hard_prediction,
                class_count=problem.class_count, metrics=metric_names))
    return result


class ProblemExtra(Enum):
    ADV = 'adv'


@dc.dataclass
class Problem:
    def get_metrics(self):
        return get_universal_metrics(), ()


@dc.dataclass
class Supervised(Problem):
    pass


@dc.dataclass
class Classification(Supervised):
    aliases = dict(class_count=['num_classes'])

    class_count: int
    extra: T.List[ProblemExtra] = dc.field(default_factory=list)

    def get_metrics(self):
        result = get_classification_metrics(self, ('A',))
        return result, ('A',)


@dc.dataclass
class SemanticSegmentation(Classification):
    shape: T.Tuple[int, int] = None  # TODO: remove " = None" and use @dc.dataclass(kw_only=True)

    def get_metrics(self):
        result = get_classification_metrics(self, ('A', 'mIoU', 'IoU'))
        return result, ('mIoU',)


@dc.dataclass
class DepthRegression(Supervised):
    shape: T.Tuple[int, int]


_name_to_class = dict(classification=Classification, semantic_segmentation=SemanticSegmentation)


def get_problem_type(name: str):
    return _name_to_class[name]


def get_problem(name: str, *args, **kwargs):
    return get_problem_type(name)(*args, **kwargs)
