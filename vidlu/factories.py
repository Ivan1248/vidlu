import warnings
import re
from functools import partial

import torch

from vidlu import defaults, models
from vidlu.data import DatasetFactory
from vidlu.data_utils import CachingDatasetFactory, DataLoader
import vidlu.modules  as m
from vidlu.problem import Supervised
from vidlu.transforms import image as imt
from vidlu.utils.func import (argtree_hard_partial, find_empty_params_deep,
                              ArgTree, params, Empty, default_args, identity, EscapedArgTree)
from vidlu.utils import tree, text

t = ArgTree  # used in arg/evaluation


# Factory messages #################################################################################

def print_all_args_message(func):
    print("All arguments:")
    print(f"Argument tree of the model ({func.func}):")
    tree.print_tree(ArgTree.from_func(func), depth=1)


def print_missing_args_message(func):
    empty_args = list(find_empty_params_deep(func))
    if len(empty_args) != 0:
        print("Unassigned arguments:")
        for ea in empty_args:
            print(f"  {'/'.join(ea)}")


def print_args_messages(kind, type_, factory, argtree, verbosity=1):
    if verbosity > 0:
        print(f'{kind}:', type_.__name__)
        tree.print_tree(argtree, depth=1)
        if verbosity > 1:
            print_all_args_message(factory)
        print_missing_args_message(factory)


# Dataset ##########################################################################################

def parse_data_str(data_str):
    def error(msg=""):
        raise ValueError(
            f'Invalid configuration string. {msg}'
            + ' Syntax: "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)')

    # full regex: r'((?:^|(?:,\s*))(\w+)(\([^{]*\))?{(\w+(?:\s*,\s*\w+)*)})'
    start_re = r'(?:^|(?:,\s*))'
    name_re, options_re, subsets_re = r'(\w+)', r'(\([^{]*\))', r'{(\w+(?:\s*,\s*\w+)*)}'
    single_ds_re = re.compile(fr'({start_re}{name_re}{options_re}?{subsets_re})')
    single_ds_configs = [x[0] for x in single_ds_re.findall(data_str)]
    reconstructed_config = ''.join(single_ds_configs)
    if reconstructed_config != data_str:
        error(f'Invalid syntax. Got "{data_str}", reconstructed as "{reconstructed_config}".')

    name_options_subsets_tuples = []
    for s in single_ds_configs:
        m = single_ds_re.match(s)
        name, options, subsets = m.groups()[1:]
        subsets = [s.strip() for s in subsets.split(',')]
        name_options_subsets_tuples += [(name, options, subsets)]
    return name_options_subsets_tuples


def get_data(data_str: str, datasets_dir, cache_dir=None):
    """

    Args:
        data_str (str): a string representing the datasets.
        datasets_dir: directory with
        cache_dir:

    Returns:

    """
    name_options_subsets_tuples = parse_data_str(data_str)

    get_parted_dataset = DatasetFactory(datasets_dir)
    if cache_dir is not None:
        get_parted_dataset = CachingDatasetFactory(get_parted_dataset, cache_dir,
                                                   add_statistics=True)
    data = dict()
    for name, options_str, subsets in name_options_subsets_tuples:
        options = eval(f'dict{options_str or "()"}')
        pds = get_parted_dataset(name, **options)
        k = f"{name}{options_str or ''}"
        if k in data:
            raise ValueError(f'"{k}" shouldn\'t occur more than once in `data_str`.')
        data[k] = {s: getattr(pds, s) for s in subsets}
    return data


get_data.help = \
    ('Dataset configuration with syntax'
     + ' "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)",'
     + ' e.g. "cifar10{train,val}", "cityscapes(remove_hood=True){trainval,val,test}",'
     + ' "inaturalist{train,all}", or "camvid{trainval}, wilddash(downsampling=2){val}"')


def get_input_preparation(dataset):
    from vidlu.transforms.input_preparation import prepare_input_image, prepare_label
    fields = tuple(dataset[0].keys())
    if fields == ('x', 'y'):
        return lambda ds: ds.map_fields(dict(x=prepare_input_image, y=prepare_label),
                                        func_name='prepare')
    raise ValueError(f"Unknown record format: {fields}.")


# Model ############################################################################################

def get_input_adapter(input_adapter_str, problem, data_statistics=None):
    """ Returns a bijective module to be inserted before the model to scale
    inputs.

    For images, it is assumed that the input elements are in range 0 to 1.

    Args:
        input_adapter_str: a string in {"standardize", "id"}
        problem (ProblemInfo):
        data_statistics (optional): a namespace containing the mean and the standard
            deviation ("std") of the training dataset.

    Returns:
        A torch module.
    """
    if isinstance(problem, Supervised):
        if input_adapter_str == "standardize":
            stats = dict(mean=torch.from_numpy(data_statistics.mean),
                         std=torch.from_numpy(data_statistics.std))
            return m.RevFunc(imt.Standardize(**stats), imt.Destandardize(**stats))
        elif input_adapter_str == "id":  # min 0, max 1 is expected for images
            return m.RevIdentity()
        else:
            raise ValueError(f"Invalid input_adapter_str: {input_adapter_str}")
    raise NotImplementedError()


# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_model(model_str: str, *, input_adapter_str='id', problem=None, init_input=None,
              dataset=None, device=None, verbosity=1):
    # imports available for evals
    import torch.nn
    from vidlu import modules as m
    from vidlu.modules import loss
    import vidlu.modules.components as C
    import torchvision.models as tvmodels

    if dataset is None and (problem is None or init_input is None):
        raise ValueError("get_model: If dataset is None, problem and init_input need to be given.")

    if problem is None:
        problem = defaults.get_problem_from_dataset(dataset)
    if init_input is None and dataset is not None:
        init_input = batch_x = next(iter(DataLoader(dataset, batch_size=2)))[0]
        init_input = dataset[0].x.unsqueeze(0)

    # `argtree_arg` has at most 1 element because `maxsplit`=1
    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    model_class = getattr(models, model_name) if hasattr(models, model_name) else eval(model_name)
    argtree = defaults.get_model_argtree(model_class, problem)
    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) == 1 else ArgTree()
    argtree.update(argtree_arg)

    model_f = argtree_hard_partial(
        model_class,
        **argtree,
        input_adapter=get_input_adapter(
            input_adapter_str,
            problem=problem,
            data_statistics=None if dataset is None else dataset.info.cache['standardization']))

    print_args_messages('Model', model_class, model_f, argtree, verbosity=verbosity)

    model = model_f()
    model.eval()

    if hasattr(model, 'initialize'):
        model.initialize(init_input)
    else:
        warnings.warn("The model does not have an initialize method.")
        model(init_input)
    if device is not None:
        model.to(device)

    return model


get_model.help = \
    ('Model configuration with syntax "Model[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees. '
     + 'Instead of ArgTree(...), t(...) can be used. '
     + 'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')

"configs.supervised,configs.classification,configs.resnet_cifar"


# Training and evaluation ##########################################################################

# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_trainer(trainer_str: str, dataset, model, verbosity=1):
    # imports available for evals
    from torch import optim
    from torch.optim import lr_scheduler
    from vidlu.training import trainers, configs
    from vidlu.training.adversarial import attacks
    from vidlu.utils.misc import fuse
    from vidlu.transforms import jitter

    trainer_name, *argtree_arg = (x.strip() for x in trainer_str.strip().split(',', 1))
    trainer_class = getattr(trainers, trainer_name)
    argtree = defaults.get_trainer_argtree(trainer_class, dataset)
    argtree_args = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) == 1 else ArgTree()
    if 'optimizer_f' in argtree_args and 'weight_decay' in default_args(argtree_args.optimizer_f):
        raise ValueError("The `weight_decay` argument should be passed to the trainer instead of"
                         + " the optimizer.")
    argtree.update(argtree_args)
    trainer_f = argtree_hard_partial(trainer_class, **argtree)

    print_args_messages('Trainer', trainer_class, trainer_f, argtree, verbosity=verbosity)

    return trainer_f(model=model)


get_trainer.help = \
    ('Trainer configuration with syntax "Trainer[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees.'
     + ' Instead of ArgTree(...), t(...) can be used.'
     + ' Example: "ResNetCifarTrainer"')

"""
def get_pretrained_params(model, params_name, params_dir):
    return defaults.get_pretrained_params(model, params_name, params_dir)
"""


# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_metrics(metrics_str: str, trainer, *, problem=None, dataset=None):
    from vidlu.training import metrics

    if problem is None:
        if dataset is None:
            raise ValueError("get_metrics: either the dataset argument"
                             + " or the problem argument need to be given.")
        problem = defaults.get_problem_from_dataset(dataset)

    default_metrics = defaults.get_metrics(trainer, problem)

    metrics_str = metrics_str.strip()
    metric_names = [x.strip() for x in metrics_str.strip().split(',')] if metrics_str else []
    additional_metrics = [getattr(metrics, name) for name in metric_names]

    def add_missing_args(metric_f):
        dargs = defaults.get_metric_args(problem)
        missing = [p for p in params(metric_f) if p is Empty]
        return (partial(metric_f, **{p: dargs[p] for p in missing}) if len(missing) > 0
                else metric_f)

    return list(map(add_missing_args, default_metrics + additional_metrics))
