import inspect
import warnings
from functools import partial
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
    args = {k: tuple(dataset[0].y.shape) if k == "y_shape" else dataset.info[k]
            for k in params(problem_type)}
    return problem_type(**args)


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
    else:
        return None
        warnings.warn(f"get_model_argtree: Unknown model type {model_class}")


# Trainer/Evaluator ################################################################################

def get_trainer_args(trainer_extension_fs, dataset):
    from vidlu.modules import losses

    problem = get_problem_from_dataset(dataset)
    args = dict()
    if isinstance(problem, (Classification, SemanticSegmentation)):
        args.update(loss=losses.NLLLossWithLogits(ignore_index=-1))
    return args


# Metrics ##########################################################################################

def get_metrics(trainer, problem):  # TODO: move to configs
    from vidlu import metrics

    common_names = ['mem', 'freq', 'loss', 'A', 'mIoU']

    ret = [partial(metrics.MaxMultiMetric, filter=lambda k, v: k.startswith('mem')),
           partial(metrics.HarmonicMeanMultiMetric, filter=lambda k, v: k.startswith('freq')),
           partial(metrics.AverageMultiMetric, filter=lambda k, v: k.startswith('loss'))]

    if isinstance(problem, (Classification, SemanticSegmentation)):
        clf_metric_names = ('A', 'mIoU', 'IoU') if isinstance(problem, SemanticSegmentation) else ('A',)
        hard_prediction_name = "other_outputs.hard_prediction"
        ret.append(partial(metrics.AverageMultiMetric,
                           filter=lambda k, v: isinstance(v, (int, float))
                                               and not (any(k.startswith(c) for c in common_names))))
        if any(isinstance(e, t.AdversarialTraining) for e in trainer.extensions):
            ret.append(partial(metrics.with_suffix(metrics.ClassificationMetrics, 'adv'),
                               hard_prediction_name="other_outputs_p.hard_prediction",
                               class_count=problem.class_count, metrics=clf_metric_names))
        elif any(isinstance(e, t.SemisupVAT) for e in trainer.extensions):
            hard_prediction_name = "other_outputs_l.hard_prediction"
        ret.append(partial(metrics.ClassificationMetrics, hard_prediction_name=hard_prediction_name,
                           class_count=problem.class_count, metrics=clf_metric_names))
        main_metrics = ("mIoU",) if isinstance(problem, SemanticSegmentation) else ("A",)
    elif isinstance(problem, DepthRegression):
        raise NotImplementedError()
        # ret = [metrics.MeanSquaredError, metrics.MeanAbsoluteError]
    else:
        raise RuntimeError(f"get_metrics: There are no default metrics for problem {type(problem)}.")
    return ret, main_metrics
