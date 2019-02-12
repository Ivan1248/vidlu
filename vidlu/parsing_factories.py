import warnings
import re
from functools import partial

import torch
import numpy as np
from torchvision.transforms.functional import to_tensor as image_to_tensor

from vidlu.data import DatasetFactory, DataLoader
from vidlu.data_utils import cache_data_and_normalize_inputs
from vidlu.utils.func import (ArgTree, argtree_partial, argtree_hard_partial,
                              find_empty_params_deep, params_deep, params, Empty, valmap)
from vidlu.utils.tree import tree_to_paths, print_tree

t = ArgTree  # used in arg/evaluation


def parse_datasets(datasets_str: str, datasets_dir, cache_dir):
    from vidlu.data import Record

    def error(msg=""):
        raise ValueError(f'Invalid configuration string. {msg} ' + \
                         'Syntax: "dataset1[([arg1=val1[, ...]])]{subset1[,...]}[, ...]".')

    pds_factory = DatasetFactory(datasets_dir)
    get_parted_dataset = lambda name, **k: cache_data_and_normalize_inputs(pds_factory(name, **k),
                                                                           cache_dir)

    datasets_str = datasets_str.strip(' ,') + ','
    single_ds_regex = re.compile(r'(\s*(\w+)(\([^{]*\))?{(\w+(?:\s*,\s*\w+)*)}\s*(?:,|\s)\s*)')
    single_ds_configs = [x[0] for x in single_ds_regex.findall(datasets_str)]
    reconstructed_config = ''.join(single_ds_configs)
    if reconstructed_config != datasets_str:
        error(f'Got "{datasets_str}", reconstructed as "{reconstructed_config}".')

    def label_to_tensor(x):
        kwargs = {}
        if x.dtype == np.int8:  # workaround as PyTorch doesn't support np.int8 -> torch.int8
            x = x.astype(np.int16)
        return torch.tensor(x, **kwargs)

    datasets = []
    for single_ds_config in single_ds_configs:
        m = single_ds_regex.match(single_ds_config)
        name, options, subsets = m.groups()[1:]
        options = eval(f'dict{options or "()"}')
        subsets = [s.strip() for s in subsets.split(',')]
        pds = get_parted_dataset(name, **options).with_transform(
            lambda ds: ds.map(lambda r: Record(x=image_to_tensor(r.x), y=label_to_tensor(r.y))))
        datasets += [getattr(pds, x) for x in subsets]

    # return [ds.map(lambda d: valmap(to_tensor, d, Record)) for ds in datasets]
    return datasets


parse_datasets.help = \
    ('Dataset configuration, e.g. "cifar10{train,val}",' +
     '"cityscapes(remove_hood=True){trainval,val,test}", "inaturalist{train,all}", or ' +
     '"camvid{trainval}, wilddash(downsampling_factor=2){val}')


def print_all_args_message(func):
    print("All arguments:")
    # for p, v in tree_to_paths(params_deep(func)):
    #    print('.'.join(p), '=', v)

    print(f"Argument tree of the model ({func.func}):")
    print_tree(params_deep(func), ArgTree, depth=1)


def print_missing_args_message(func):
    empty_args = find_empty_params_deep(func)
    if len(empty_args) != 0:
        print("Unassigned arguments:")
        for ea in empty_args:
            print(f"  {'/'.join(ea)}")


def print_args_messages(kind, type_, factory, argtree, verbosity=1):
    if verbosity > 0:
        print(f'{kind}:', type_.__name__)
        print_tree(argtree, depth=1)
        if verbosity > 1:
            print_all_args_message(factory)
        print_missing_args_message(factory)


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_model(model_str: str, dataset, device=None, verbosity=1):
    import torch.nn  # used in model_str/evaluation
    import vidlu.nn  # used in model_str/evaluation
    from vidlu.nn import components
    from vidlu import models
    import torchvision as tv

    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))
    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) > 0 else ArgTree()
    try:
        model_class = getattr(models, model_name)
    except:
        model_class = eval(model_name)
    argtree = models.get_default_argtree(model_class, dataset)
    argtree.update(argtree_arg)
    model_f = argtree_hard_partial(model_class, **argtree)

    print_args_messages('Model', model_class, model_f, argtree, verbosity=verbosity)

    model = model_f()
    batch_x = next(iter(DataLoader(dataset)))[0]
    try:
        model.initialize(batch_x)
    except AttributeError:
        warnings.warn("The model doesn't have an initialize method.")
    return model


parse_model.help = \
    ('Model defined in format Model[,arg=value,...], where values can be ArgTrees. ' +
     'Instead of ArgTree(...), t(...) can be used. ' +
     'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_trainer(trainer_str: str, model, dataset, device=None, verbosity=1):
    from torch import optim
    from torch.optim import lr_scheduler
    from vidlu import trainers

    trainer_name, *argtree_arg = (x.strip() for x in trainer_str.strip().split(',', 1))
    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) > 0 else ArgTree()
    trainer_class = getattr(trainers, trainer_name)

    argtree = trainers.get_default_argtree(trainer_class, model, dataset)
    argtree.update(argtree_arg)
    trainer_f = argtree_hard_partial(trainer_class, **argtree)

    print_args_messages('Trainer', trainer_class, trainer_f, argtree, verbosity=verbosity)

    return trainer_f(model=model, device=device)


parse_trainer.help = \
    ('Trainer defined in format Trainer[,arg=value,...], where values can be ArgTrees. ' +
     'Instead of ArgTree(...), t(...) can be used. ' +
     'Example: "ResNetCifarTrainer"')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_metrics(metrics_str: str, dataset):
    from vidlu import metrics

    default_metrics = metrics.get_default_metrics(dataset)

    metrics_str = metrics_str.strip()
    metric_names = [x.strip() for x in metrics_str.strip().split(',')] if metrics_str else []
    additional_metrics = [getattr(metrics, name) for name in metric_names]

    def add_missing_args(metric_f):
        dargs = metrics.get_default_metric_args(dataset)
        missing = [p for p in params(metric_f) if p is Empty]
        return (partial(metric_f, **{p: dargs[p] for p in missing}) if len(missing) > 0
                else metric_f)

    metric_fs = set(default_metrics + additional_metrics)
    return set(map(add_missing_args, metric_fs))


def parse_pretrained_params(pretrained_params_str):
    pass


# run.py data model trainer evaluation
""" python run.py 
        cifar10{trainval,test} 
        "ResNet,backbone_f=t(depth=34, small_input=True))"
        "ResNetCifarTrainer"
        "accuracy"
"""
