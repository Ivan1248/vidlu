import warnings
import re
from functools import partial

from vidlu import defaults, models
from vidlu.utils.func import (argtree_hard_partial, find_empty_params_deep,
                              ArgTree, params, Empty)
from vidlu.utils.tree import print_tree

t = ArgTree  # used in arg/evaluation


# Factory messages #################################################################################

def print_all_args_message(func):
    print("All arguments:")
    print(f"Argument tree of the model ({func.func}):")
    print_tree(ArgTree.from_func(func), depth=1)


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


# Dataset ##########################################################################################

def parse_datasets(datasets_str: str, datasets_dir, cache_dir=None):
    from vidlu.data import DatasetFactory
    from vidlu.data_utils import CachingDatasetFactory

    def error(msg=""):
        raise ValueError(
            f'Invalid configuration string. {msg}'
            + ' Syntax: "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)')

    datasets_str = datasets_str.strip(' ,') + ','
    single_ds_regex = re.compile(r'(\s*(\w+)(\([^{]*\))?{(\w+(?:\s*,\s*\w+)*)}\s*(?:,|\s)\s*)')
    single_ds_configs = [x[0] for x in single_ds_regex.findall(datasets_str)]
    reconstructed_config = ''.join(single_ds_configs)
    if reconstructed_config != datasets_str:
        error(f'Got "{datasets_str}", reconstructed as "{reconstructed_config}".')

    name_options_subsets_tuples = []
    for s in single_ds_configs:
        m = single_ds_regex.match(s)
        name, options, subsets = m.groups()[1:]
        subsets = [s.strip() for s in subsets.split(',')]
        name_options_subsets_tuples += [(name, options, subsets)]

    get_parted_dataset = (DatasetFactory(datasets_dir) if cache_dir is None
                          else CachingDatasetFactory(datasets_dir, cache_dir))
    datasets = []
    for name, options, subsets in name_options_subsets_tuples:
        options = eval(f'dict{options or "()"}')
        pds = get_parted_dataset(name, **options)
        datasets += [getattr(pds, x) for x in subsets]

    return datasets


parse_datasets.help = \
    ('Dataset configuration with syntax'
     + ' "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)",'
     + ' e.g. "cifar10{train,val}", "cityscapes(remove_hood=True){trainval,val,test}",'
     + ' "inaturalist{train,all}", or "camvid{trainval}, wilddash(downsampling_factor=2){val}"')


# Model ############################################################################################

# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_model(model_str: str, dataset, device=None, verbosity=1):
    import torch.nn  # used in model_str/evaluation
    from vidlu import modules  # used in model_str/evaluation
    from vidlu.modules import loss
    import vidlu.modules.components as com
    import torchvision.models as tvmodels
    from vidlu.data import DataLoader

    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) > 0 else ArgTree()
    try:
        model_class = getattr(models, model_name)
    except:
        model_class = eval(model_name)
    argtree = defaults.get_model_argtree(model_class, dataset)
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
    return model


parse_model.help = \
    ('Model configuration with syntax "Model[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees. '
     + 'Instead of ArgTree(...), t(...) can be used. '
     + 'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')


# Trainer and metrics ##############################################################################

# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_trainer(trainer_str: str, model, dataset, device=None, verbosity=1):
    from torch import optim
    from torch.optim import lr_scheduler
    from vidlu.training import trainers

    trainer_name, *argtree_arg = (x.strip() for x in trainer_str.strip().split(',', 1))

    argtree_arg = eval(f"ArgTree({argtree_arg[0]})") if len(argtree_arg) > 0 else ArgTree()
    if 'optimizer_f' in argtree_arg and 'weight_decay' in argtree_arg.optimizer_f:
        raise ValueError("weight_decay is allowed to be given as an argument to the trainer,"
                         + " but not to the optimizer.")
    trainer_class = getattr(trainers, trainer_name)

    argtree = defaults.get_trainer_argtree(trainer_class, model, dataset)
    argtree.update(argtree_arg)
    trainer_f = argtree_hard_partial(trainer_class, **argtree)

    print_args_messages('Trainer', trainer_class, trainer_f, argtree, verbosity=verbosity)

    return trainer_f(model=model, device=device)


parse_trainer.help = \
    ('Trainer configuration with syntax "Trainer[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees.'
     + ' Instead of ArgTree(...), t(...) can be used.'
     + ' Example: "ResNetCifarTrainer"')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_metrics(metrics_str: str, dataset):
    from vidlu.training import metrics

    default_metrics = defaults.get_metrics(dataset)

    metrics_str = metrics_str.strip()
    metric_names = [x.strip() for x in metrics_str.strip().split(',')] if metrics_str else []
    additional_metrics = [getattr(metrics, name) for name in metric_names]

    def add_missing_args(metric_f):
        dargs = defaults.get_metric_args(dataset)
        missing = [p for p in params(metric_f) if p is Empty]
        return (partial(metric_f, **{p: dargs[p] for p in missing}) if len(missing) > 0
                else metric_f)

    return list(map(add_missing_args, default_metrics + additional_metrics))


def parse_pretrained_params(pretrained_params_str):
    pass
