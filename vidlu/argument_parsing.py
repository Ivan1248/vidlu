import vidlu
import re


def parse_datasets_arg(arg: str):
    from vidlu.data_utils import get_cached_dataset_set_with_normalized_inputs

    def error(msg=""):
        raise ValueError(f'Invalid configuration string. {msg} ' + \
                         'Syntax: "dataset1[([arg1=val1[, ...]])]{subset1[,...]}[, ...]".')

    arg = arg.strip(' ,') + ','
    single_ds_regex = re.compile(r'(\s*(\w+)(\([^{]*\))?{(\w+(?:\s*,\s*\w+)*)}\s*(?:,|\s)\s*)')
    single_ds_configs = [x[0] for x in single_ds_regex.findall(arg)]
    reconstructed_config = ''.join(single_ds_configs)
    if reconstructed_config != arg:
        error(f'Got "{arg}", reconstructed as "{reconstructed_config}".')

    datasets = []
    for single_ds_config in single_ds_configs:
        m = single_ds_regex.match(single_ds_config)
        name, options, subsets = m.groups()[1:]
        options = eval(f'dict{options or "()"}')
        subsets = [s.strip() for s in subsets.split(',')]
        ds = get_cached_dataset_set_with_normalized_inputs(name, **eval(f'dict{options or "()"}'))
        datasets += [getattr(ds, x) for x in subsets]

    return datasets


parse_datasets_arg.help = \
    ('Dataset configuration, e.g. "cifar10{train,val}",' +
     '"cityscapes(remove_hood=True){trainval,val,test}", "inaturalist{train,all}", or ' +
     '"camvid{trainval}, wilddash(downsampling_factor=2){val}')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_model_arg(arg: str):
    from vidlu.learning import models
    from vidlu.utils.func import ArgTree, argtree_partial, find_empty_args
    import torch.nn as nn  # used in arg/eval
    import vidlu.nn as vnn  # used in arg/eval
    a = ArgTree  # used in arg/eval
    name, argtree = (x.strip() for x in arg.strip().split(',', 1))
    model_f = getattr(models, name)
    argtree = eval(f"dict({argtree})")
    model_f = argtree_partial(model_f, **argtree)
    for a in find_empty_args(model_f, vidlu):
        print(f"Unassigned: {a}")
    return name, model_f


parse_datasets_arg.help = \
    ('Model defined in format Model[,arg=value,...], where values can be ArgTrees. ' +
     'Instead of ArgTree(...), a(...) can be used. ' +
     'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')


# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_training_arg(arg: str):
    from vidlu.learning import training_configs
    from vidlu.utils.func import ArgTree, argtree_partial, find_empty_args
    import torch.nn  # used in arg/eval
    import vidlu.nn  # used in arg/eval
    from torch import optim
    from torch.optim import lr_scheduler
    a = ArgTree  # used in arg/eval
    name, argtree = (x.strip() for x in arg.strip().split(',', 1))
    training_config_f = getattr(training_configs, name)
    argtree = eval(f"dict({argtree})")
    training_config_f = argtree_partial(training_config_f, **argtree)
    for a in find_empty_args(training_config_f, vidlu):
        print(f"Unassigned: {a}")
    return name, training_config_f

# noinspection PyUnresolvedReferences,PyUnusedLocal
def parse_evaluation_arg(arg: str):
    pass
# run.py data model training metrics
""" run.py 
        cifar10{trainval,test} 
        "ResNet,backbone_f=a(depth=34, base_width=64, small_input=True))"
        "ResNetCifar"
        "accuracy"
"""


def parse_training_arg(arg: str):
    pass
