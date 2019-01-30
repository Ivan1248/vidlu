import re
# noinspection PyUnresolvedReferences
from functools import partial
from vidlu.utils.func import ArgTree, argtree_hard_partial, find_empty_args
from vidlu.problems import Problems, dataset_to_problem

t = ArgTree  # used in arg/eval


def parse_datasets(datasets_str: str, get_dataset):
    def error(msg=""):
        raise ValueError(f'Invalid configuration string. {msg} ' + \
                         'Syntax: "dataset1[([arg1=val1[, ...]])]{subset1[,...]}[, ...]".')

    datasets_str = datasets_str.strip(' ,') + ','
    single_ds_regex = re.compile(r'(\s*(\w+)(\([^{]*\))?{(\w+(?:\s*,\s*\w+)*)}\s*(?:,|\s)\s*)')
    single_ds_configs = [x[0] for x in single_ds_regex.findall(datasets_str)]
    reconstructed_config = ''.join(single_ds_configs)
    if reconstructed_config != datasets_str:
        error(f'Got "{datasets_str}", reconstructed as "{reconstructed_config}".')

    datasets = []
    for single_ds_config in single_ds_configs:
        m = single_ds_regex.match(single_ds_config)
        name, options, subsets = m.groups()[1:]
        options = eval(f'dict{options or "()"}')
        subsets = [s.strip() for s in subsets.split(',')]
        ds = get_dataset(name, **eval(f'dict{options or "()"}'))
        datasets += [getattr(ds, x) for x in subsets]

    return datasets


parse_datasets.help = \
    ('Dataset configuration, e.g. "cifar10{train,val}",' +
     '"cityscapes(remove_hood=True){trainval,val,test}", "inaturalist{train,all}", or ' +
     '"camvid{trainval}, wilddash(downsampling_factor=2){val}')


def print_missing_args_message(func):
    empty_args = find_empty_args(func)
    if len(empty_args) != 0:
        print("Unassigned arguments:")
        for ea in empty_args:
            print(f"  {ea}")


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_model(model_str: str, dataset):
    import torch.nn  # used in model_str/eval
    import vidlu.nn  # used in model_str/eval
    from vidlu.nn import components
    from vidlu import models

    model_name, argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))
    argtree_arg = eval(f"ArgTree({argtree_arg})")
    model_class = getattr(models, model_name)
    argtree = models.get_default_argtree(model_class, dataset)
    argtree.update(argtree_arg)
    model_f = argtree_hard_partial(model_class, **argtree)

    print_missing_args_message(model_f)

    return model_name, model_f


parse_datasets.help = \
    ('Model defined in format Model[,arg=value,...], where values can be ArgTrees. ' +
     'Instead of ArgTree(...), a(...) can be used. ' +
     'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_training_config(training_str: str):
    import training_configs
    from torch import optim
    from torch.optim import lr_scheduler

    name, argtree = (x.strip() for x in training_str.strip().split(',', 1))
    training_config_f = getattr(training_configs, name)
    argtree = eval(f"dict({argtree})")
    training_config_f = argtree_hard_partial(training_config_f, **argtree)

    print_missing_args_message(training_config_f)

    return name, training_config_f


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_evaluation_arg(arg: str):
    pass


# run.py pds model training metrics
""" python run.py 
        cifar10{trainval,test} 
        "ResNet,backbone_f=t(depth=34, small_input=True))"
        "ResNetCifar"
        "accuracy"
"""
