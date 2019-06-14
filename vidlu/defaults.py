import inspect
import warnings
from functools import partial

from ignite import metrics

from vidlu.data import Record
from vidlu.models import DiscriminativeModel, Autoencoder
from vidlu.modules import components
from vidlu.modules import loss
from vidlu.problem import Problem
from vidlu.training.metrics import FuncMetric, ClassificationMetrics, ClassificationMetricsAdv
from vidlu.training.trainers import Trainer, AdversarialTrainer
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
                    head_f=partial(components.ClassificationHead, class_count=dataset.info.class_count))
            elif problem == Problem.SEMANTIC_SEGMENTATION:
                return ArgTree(
                    head_f=partial(components.SegmentationHead, class_count=dataset.info.class_count,
                                   shape=dataset[0].y.shape))
            elif problem == Problem.DEPTH_REGRESSION:
                return ArgTree(
                    head_f=partial(components.RegressionHead, shape=dataset[0].y.shape))
            elif problem == Problem.OTHER:
                return ArgTree()
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if problem == Problem.CLASSIFICATION:
            return ArgTree(num_classes=dataset.info.class_count)
    warnings.warn(f"get_default_argtree: Unknown model type {model_class}")
    return ArgTree()


# Training data jittering and model input perparation ##############################################

def get_jitter(dataset):
    from vidlu.transforms.jitter import cifar_jitter, rand_hflip
    if any(dataset.name.lower().startswith(x) for x in ['cifar', 'cifar100', 'tinyimagenet']):
        return lambda r: Record(x=cifar_jitter(r.x), y=r.y)
    elif dataset.info['problem'] == 'semseg':
        return lambda r: Record(zip(['x', 'y'], rand_hflip(r.x, r.y)))
    else:
        return lambda x: x


# Trainer/Evaluator ################################################################################

def get_trainer_argtree(trainer_class, dataset):
    problem = get_problem(dataset)
    argtree = ArgTree()
    if issubclass(trainer_class, Trainer):
        if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
            argtree.update(ArgTree(loss_f=partial(loss.SoftmaxCrossEntropyLoss, ignore_index=-1)))
        if issubclass(trainer_class, AdversarialTrainer):
            pass  # it will be ovderridden anyway with the attack
            # if 'map_div255' not in dataset.modifiers:
            #    raise RuntimeError(
            #        "Adversarial attacks supported only for element values from interval [0, 1].")
            # argtree.update(attack_f=ArgTree(clip_bounds=(0, 1)))
    return argtree


# Metrics ##########################################################################################

def get_metrics(trainer, dataset):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        ret = [partial(FuncMetric, func=lambda iter_output: iter_output.loss, name='loss'),
               partial(ClassificationMetrics, class_count=dataset.info.class_count)]
        if isinstance(trainer, AdversarialTrainer):
            ret.append(partial(ClassificationMetricsAdv, class_count=dataset.info.class_count))
    elif problem == Problem.DEPTH_REGRESSION:
        ret = [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    elif problem == Problem.OTHER:
        ret = []
    return ret


def get_metric_args(dataset):
    problem = get_problem(dataset)
    if problem in [Problem.CLASSIFICATION, Problem.SEMANTIC_SEGMENTATION]:
        return dict(class_count=dataset.info.class_count)
    elif problem == Problem.DEPTH_REGRESSION:
        return {}
    elif problem == Problem.OTHER:
        return {}
