import inspect
import warnings
from functools import partial

from ignite import metrics
from torch import nn

from vidlu.nn.models import DiscriminativeModel, Autoencoder
from vidlu.nn import components as c
from vidlu.problem import Problem
from vidlu.training.metrics import FuncMetric, ClassificationMetrics
from vidlu.training.trainers import SupervisedTrainer
from vidlu.utils.func import ArgTree


# Problem ######################################################################

def get_problem(dataset):
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


# Model ############################################################################################

def get_model_argtree(model_class, dataset):
    problem = get_problem(dataset)
    if inspect.isclass(model_class):
        if issubclass(model_class, DiscriminativeModel):
            if problem == Problem.CLASSIFICATION:
                return ArgTree(head_f=partial(c.ClassificationHead,
                                              class_count=dataset.info.class_count))
            elif problem == Problem.SEMANTIC_SEGMENTATION:
                return ArgTree(head_f=partial(c.SegmentationHead,
                                              class_count=dataset.info.class_count,
                                              shape=dataset[0].y.shape[1:]))
            elif problem == Problem.DEPTH_REGRESSION:
                return ArgTree(head_f=partial(c.RegressionHead,
                                              shape=dataset[0].y.shape[1:]))
            elif problem == Problem.OTHER:
                return ArgTree()
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if problem == Problem.CLASSIFICATION:
            return ArgTree(num_classes=dataset.info.class_count)
    warnings.warn(f"get_default_argtree: Unknown model type {model_class}")
    return ArgTree()


# Trainer ##########################################################################################

def get_trainer_argtree(trainer_class, model, dataset):
    problem = get_problem(dataset)
    if issubclass(trainer_class, SupervisedTrainer):
        if problem == Problem.CLASSIFICATION:
            return ArgTree(loss_f=nn.NLLLoss)
        elif problem == Problem.SEMANTIC_SEGMENTATION:
            return ArgTree(loss_f=nn.NLLLoss)
    return ArgTree()


# Metrics ##########################################################################################

def get_default_metrics(dataset):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        return [partial(FuncMetric, func=lambda iter_output: iter_output.loss, name='loss'),
                partial(ClassificationMetrics, class_count=dataset.info.class_count)]
    elif problem == Problem.DEPTH_REGRESSION:
        return [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    elif problem == Problem.OTHER:
        return []


def get_default_metric_args(dataset):
    problem = get_problem(dataset)
    if problem == Problem.CLASSIFICATION:
        return dict(class_count=dataset.info.class_count)
    elif problem == Problem.SEMANTIC_SEGMENTATION:
        return dict(class_count=dataset.info.class_count)
    elif problem == Problem.DEPTH_REGRESSION:
        return dict()
    elif problem == Problem.OTHER:
        return dict()
