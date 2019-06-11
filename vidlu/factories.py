import warnings
import re
from dataclasses import dataclass
from functools import partial, wraps

import torch
import dill

from vidlu import defaults, models
from vidlu.data import Record, Dataset, DatasetFactory
from vidlu.data_utils import CachingDatasetFactory
from vidlu.transforms import image as T
from vidlu.utils.collections import NameDict
from vidlu.utils.func import (argtree_hard_partial, find_empty_params_deep,
                              ArgTree, params, Empty, default_args, functree, identity)
from vidlu.utils import tree

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
        datasets_str (str): a string representing the datasets.
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


def get_input_preparation(input_prep_str, dataset):
    from vidlu.transforms.input_preparation import prepare_input_label
    fields = tuple(dataset[0].keys())
    if fields == ('x', 'y'):
        if input_prep_str == "standardize":
            stand = dataset.info.cache['standardization']
            transform = T.Standardize(mean=torch.from_numpy(stand.mean),
                                      std=torch.from_numpy(stand.std))
        elif input_prep_str == "div255":
            transform = T.Div(255)
        elif input_prep_str == "nop":
            transform = identity
        return lambda ds: ds.map_fields(dict(
            x=T.Compose(T.ToTorch(), T.HWCToCHW(), T.To(dtype=torch.float), transform),
            y=prepare_input_label), func_name=input_prep_str)
    raise ValueError(f"Unknown record format: {fields}.")


# Model ############################################################################################

# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_model(model_str: str, dataset, device=None, verbosity=1):
    import torch.nn  # used in model_str/evaluation
    from vidlu import modules  # used in model_str/evaluation
    from vidlu.modules import loss
    import vidlu.modules.components as com
    import torchvision.models as tvmodels
    from vidlu.data_utils import DataLoader

    # `argtree_arg` has at most 1 element because `maxsplit`=1
    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    model_class = getattr(models, model_name) if hasattr(models, model_name) else eval(model_name)
    argtree = defaults.get_model_argtree(model_class, dataset)
    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) == 1 else ArgTree()
    argtree.update(argtree_arg)
    model_f = argtree_hard_partial(model_class, **argtree)

    print_args_messages('Model', model_class, model_f, argtree, verbosity=verbosity)

    model = model_f()
    model.eval()
    batch_x = next(iter(DataLoader(dataset, batch_size=2)))[0]
    if hasattr(model, 'initialize'):
        model.initialize(batch_x)
    else:
        model(batch_x)
        warnings.warn("The model does not have an initialize method.")
    if device is not None:
        model.to(device)
    return model


get_model.help = \
    ('Model configuration with syntax "Model[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees. '
     + 'Instead of ArgTree(...), t(...) can be used. '
     + 'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')

"configs.supervised,configs.classification,configs.resnet_cifar"


# Trainer and metrics ##############################################################################

# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_trainer(trainer_str: str, dataset, model, verbosity=1):
    from torch import optim
    from torch.optim import lr_scheduler
    from vidlu.training import trainers, configs
    from vidlu.training.adversarial import attacks
    from vidlu.utils.misc import fuse

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


# noinspection PyUnresolvedReferences,PyUnusedLocal
def get_metrics(metrics_str: str, trainer, dataset):
    from vidlu.training import metrics

    default_metrics = defaults.get_metrics(trainer, dataset)

    metrics_str = metrics_str.strip()
    metric_names = [x.strip() for x in metrics_str.strip().split(',')] if metrics_str else []
    additional_metrics = [getattr(metrics, name) for name in metric_names]

    def add_missing_args(metric_f):
        dargs = defaults.get_metric_args(dataset)
        missing = [p for p in params(metric_f) if p is Empty]
        return (partial(metric_f, **{p: dargs[p] for p in missing}) if len(missing) > 0
                else metric_f)

    return list(map(add_missing_args, default_metrics + additional_metrics))


def get_pretrained_params(pretrained_params_str):
    pass
