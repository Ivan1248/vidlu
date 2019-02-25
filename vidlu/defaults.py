import inspect
import warnings
from functools import partial

from ignite import metrics
from torch import nn

from vidlu.data import Record
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


# Model ############################################################################################


def get_model_argtree(model_class, dataset):
    problem = get_problem(dataset)
    if inspect.isclass(model_class):
        if issubclass(model_class, DiscriminativeModel):
            if problem == Problem.CLASSIFICATION:
                return ArgTree(
                    head_f=partial(
                        c.ClassificationHead,
                        class_count=dataset.info.class_count))
            elif problem == Problem.SEMANTIC_SEGMENTATION:
                return ArgTree(
                    head_f=partial(
                        c.SegmentationHead,
                        class_count=dataset.info.class_count,
                        shape=dataset[0].y.shape))
            elif problem == Problem.DEPTH_REGRESSION:
                return ArgTree(
                    head_f=partial(c.RegressionHead, shape=dataset[0].y.shape))
            elif problem == Problem.OTHER:
                return ArgTree()
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if problem == Problem.CLASSIFICATION:
            return ArgTree(num_classes=dataset.info.class_count)
    warnings.warn(f"get_default_argtree: Unknown model type {model_class}")
    return ArgTree()


# Taining data jittering and model input perparation ###############################################


def get_jitter(dataset):
    from vidlu.transforms.jitter import cifar_jitter, rand_hflip
    if any(dataset.name.lower().startswith(x)
           for x in ['cifar', 'cifar100', 'tinyimagenet']):
        return lambda r: Record(x=cifar_jitter(r.x), y=r.y)
    elif dataset.info['problem'] == 'semseg':
        return lambda r: Record(zip(['x', 'y'], rand_hflip(r.x, r.y)))
    else:
        return lambda x: x


def get_input_preparation(dataset):
    from vidlu.transforms.input_preparation import prepare_input_image, prepare_input_label
    if 'standardization' in dataset.info:
        stand = dataset.info.standardization
        return lambda r: Record(x=prepare_input_image(r.x, mean=stand.mean, std=stand.std),
                                y=prepare_input_label(r.y))
    else:
        raise ValueError("Pixel mean and standard deviation for image datasets"
                         + " should be in dataset.info.standardization.")


# Trainer/Evaluator ################################################################################


def get_trainer_argtree(trainer_class, model, dataset):
    problem = get_problem(dataset)
    argtree = ArgTree()
    if issubclass(trainer_class, SupervisedTrainer):
        if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
            argtree.update(ArgTree(loss_f=partial(nn.NLLLoss, ignore_index=-1)))
    return argtree


# Metrics ##########################################################################################


def get_metrics(dataset):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        return [
            partial(
                FuncMetric,
                func=lambda iter_output: iter_output.loss,
                name='loss'),
            partial(
                ClassificationMetrics, class_count=dataset.info.class_count)
        ]
    elif problem == Problem.DEPTH_REGRESSION:
        return [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    elif problem == Problem.OTHER:
        return []


def get_metric_args(dataset):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        return dict(class_count=dataset.info.class_count)
    elif problem == Problem.DEPTH_REGRESSION:
        return dict()
    elif problem == Problem.OTHER:
        return dict()
