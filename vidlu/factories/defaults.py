import inspect
import warnings
from vidlu.utils.func import partial
import logging

from vidlu.models import DiscriminativeModel, Autoencoder, MNISTNet
from vidlu.factories.problem import (Classification, SemanticSegmentation, DepthRegression,
                                     get_problem_type)
import vidlu.training as t
from vidlu.utils.func import ArgTree, params

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Problem ######################################################################

def get_problem_from_dataset(dataset):
    if 'problem' not in dataset.info:
        raise ValueError("Unknown problem.")
    problem_type = get_problem_type(dataset.info.problem)
    if problem_type is SemanticSegmentation:
        args = {k: tuple(dataset[0].seg_map.shape) if k == "y_shape" else dataset.info[k]
                for k in params(problem_type)}
        return problem_type(**args)
    else:
        return problem_type()


# Model ############################################################################################

def get_model_argtree_for_problem(model_class, problem):
    from vidlu.modules import components
    
    if inspect.isclass(model_class):
        if issubclass(model_class, DiscriminativeModel):
            if type(problem) is Classification:
                if model_class in [MNISTNet]:
                    return ArgTree(head_f=partial(components.ClassificationHead1D,
                                                  class_count=problem.class_count))
                return ArgTree(head_f=partial(components.ClassificationHead,
                                              class_count=problem.class_count))
            elif type(problem) is SemanticSegmentation:
                if "head_f" not in params(model_class):
                    logger.info('The model factory does not accept an "head_f" argument.')
                elif not isinstance(params(model_class).head_f, components.SegmentationHead):
                    return ArgTree(head_f=partial(components.SegmentationHead,
                                                  class_count=problem.class_count,
                                                  shape=problem.y_shape))
            elif type(problem) is DepthRegression:
                return ArgTree(head_f=partial(components.RegressionHead, shape=problem.y_shape))
            else:
                raise ValueError("Invalid problem type.")
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if isinstance(problem, Classification):
            return ArgTree(num_classes=problem.class_count)
    warnings.warn(f"get_model_argtree: Unknown model type {model_class}")
    return ArgTree()


# Trainer/Evaluator ################################################################################

def get_trainer_args(dataset):
    return dict()


# Metrics ##########################################################################################

def get_metrics(trainer, problem):  # TODO: move to configs
    from vidlu import metrics

    common_names = ['mem', 'freq', 'loss', 'A', 'mIoU']

    ret = [partial(metrics.MaxMultiMetric, filter=lambda k, v: k.startswith('mem')),
           partial(metrics.HarmonicMeanMultiMetric, filter=lambda k, v: k.startswith('freq')),
           partial(metrics.AverageMultiMetric, filter=lambda k, v: k.startswith('loss'))]

    if isinstance(problem, (Classification, SemanticSegmentation)):
        clf_metric_names = ('A', 'mIoU', 'IoU') if isinstance(problem, SemanticSegmentation) else (
            'A',)
        get_hard_prediction = lambda r: r.out.argmax(1)
        ret.append(partial(metrics.AverageMultiMetric,
                           filter=lambda k, v: isinstance(v, (int, float))
                                               and not (
                               any(k.startswith(c) for c in common_names))))
        if "Adversarial" in type(trainer.train_step).__name__:
            ret.append(partial(metrics.with_suffix(metrics.ClassificationMetrics, 'adv'),
                               get_hard_prediction=get_hard_prediction,
                               class_count=problem.class_count, metrics=clf_metric_names))
        ret.append(partial(metrics.ClassificationMetrics, get_hard_prediction=get_hard_prediction,
                           class_count=problem.class_count, metrics=clf_metric_names))
        main_metrics = ("mIoU",) if isinstance(problem, SemanticSegmentation) else ("A",)
    elif isinstance(problem, DepthRegression):
        raise NotImplementedError()
        # ret = [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    else:
        raise RuntimeError(
            f"get_metrics: There are no default metrics for problem {type(problem)}.")
    return ret, main_metrics
