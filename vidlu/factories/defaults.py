import inspect
import warnings
import logging

from vidlu.models import DiscriminativeModel, Autoencoder
from vidlu.factories.problem import (Classification, SemanticSegmentation, DepthRegression,
                                     get_problem_type, Problem, ProblemExtra)
from vidlu.utils.func import ArgTree, params

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Problem ##########################################################################################

def get_problem(dataset, trainer=None) -> Problem:
    if 'problem' not in dataset.info:
        raise ValueError("Unknown problem.")
    problem_type = get_problem_type(dataset.info.problem)

    kwargs = dict()
    if problem_type is SemanticSegmentation:
        kwargs["shape"] = tuple(dataset[0].seg_map.shape)
    kwargs.update({k: dataset.info[k] for k in inspect.signature(problem_type).parameters.keys()
                   if k not in kwargs and k != 'extra'})
    kwargs['extra'] = []
    if trainer is not None and 'Adversarial' in type(trainer.train_step).__name__:
        kwargs['extra'].append(ProblemExtra.ADV)
    return problem_type(**kwargs)


# Model ############################################################################################

def get_model_argtree_for_problem(model_class, problem):
    from vidlu.modules import components

    if inspect.isclass(model_class):
        if issubclass(model_class, DiscriminativeModel):
            if type(problem) is Classification:
                return ArgTree(head_f=ArgTree(class_count=problem.class_count))
            elif type(problem) is SemanticSegmentation:
                if (head_f := params(model_class).get('head_f', None)) is None:
                    logger.info('The model factory does not accept a "head_f" argument.')
                    return ArgTree()
                else:
                    args = params(head_f)
                    new_args = dict()
                    for k in ['class_count'] + problem.aliases.get('class_count', ()):
                        if k in args:
                            new_args[k] = problem.class_count
                    if 'shape' in args:
                        new_args['shape'] = problem.shape
                    return ArgTree(head_f=ArgTree(**new_args))
            elif type(problem) is DepthRegression:
                return ArgTree(head_f=ArgTree(shape=problem.shape))
            else:
                raise ValueError("Invalid problem type.")
        elif issubclass(model_class, Autoencoder):
            return ArgTree()
    elif model_class.__module__.startswith('torchvision.models'):
        if isinstance(problem, Classification):
            return ArgTree(num_classes=problem.class_count)
    warnings.warn(f"get_model_argtree: Unknown model type {model_class}")
    return ArgTree()
