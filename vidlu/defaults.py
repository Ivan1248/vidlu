import inspect
import warnings
from functools import partial
import numpy as np

from vidlu.models import DiscriminativeModel, Autoencoder

from vidlu.problem import (Classification, SemanticSegmentation, DepthRegression,
                           get_problem_type)
from vidlu.training.trainers import Trainer, AdversarialTrainer
from vidlu.utils.func import ArgTree, params


# Problem ######################################################################

def get_problem_from_dataset(dataset):
    if 'problem' not in dataset.info:
        raise ValueError("Unknown problem.")
    problem_type = get_problem_type(dataset.info.problem)

    def get_attribute(k):
        if k in dataset.info:
            return dataset.info[k]
        elif k == 'y_shape':
            return tuple(dataset[0].y.shape)
        else:
            raise KeyError()

    args = {k: get_attribute(k) for k in params(problem_type)}
    return problem_type(**args)


# Model ############################################################################################

def get_model_argtree(model_class, problem):
    from vidlu.modules import components

    if inspect.isclass(model_class):
        if issubclass(model_class, DiscriminativeModel):
            if type(problem) is Classification:
                return ArgTree(
                    head_f=partial(components.ClassificationHead,
                                   class_count=problem.class_count))
            elif type(problem) is SemanticSegmentation and not isinstance(
                    params(model_class).head_f, components.SegmentationHead):
                return ArgTree(
                    head_f=partial(components.SegmentationHead,
                                   class_count=problem.class_count,
                                   shape=problem.y_shape))
            elif type(problem) is DepthRegression:
                return ArgTree(
                    head_f=partial(components.RegressionHead, shape=problem.y_shape))
            else:
                raise ValueError("Invalid problem type.")
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if isinstance(problem, Classification):
            return ArgTree(num_classes=problem.class_count)
    else:
        raise ValueError(f"get_model_argtree: Unknown model type {model_class}")


# Trainer/Evaluator ################################################################################

def get_trainer_argtree(trainer_class, dataset):
    from vidlu.modules import loss

    problem = get_problem_from_dataset(dataset)
    argtree = ArgTree()
    if issubclass(trainer_class, Trainer):
        if isinstance(problem, (Classification, SemanticSegmentation)):
            argtree.update(ArgTree(loss_f=partial(loss.SoftmaxCrossEntropyLoss, ignore_index=-1)))
        if issubclass(trainer_class, AdversarialTrainer):
            pass  # it will be ovderridden anyway with the attack
            # if 'map_div255' not in dataset.modifiers:
            #    raise RuntimeError(
            #        "Adversarial attacks supported only for element values from interval [0, 1].")
            # argtree.update(attack_f=ArgTree(clip_bounds=(0, 1)))
    return argtree


# Metrics ##########################################################################################

def get_metrics(trainer, problem):
    from ignite import metrics
    from vidlu.training.metrics import (FuncMetric, ClassificationMetrics,
                                        ClassificationMetricsAdversarial)

    if isinstance(problem, (Classification, SemanticSegmentation)):
        ret = [partial(FuncMetric, func=lambda iter_output: iter_output.loss, name='loss'),
               partial(ClassificationMetrics, class_count=problem.class_count)]
        if isinstance(trainer, AdversarialTrainer):
            ret.append(partial(ClassificationMetricsAdversarial, class_count=problem.class_count))
    elif isinstance(problem, DepthRegression):
        ret = [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    else:
        warnings.warn(f"get_metrics: There are no default metrics for problem {type(problem)}.")
        ret = []
    return ret


def get_metric_args(problem):
    if isinstance(problem, (Classification, SemanticSegmentation)):
        return dict(class_count=problem.class_count)
    elif isinstance(problem, DepthRegression):
        return {}
    return {}
