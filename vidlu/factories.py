import warnings
import re
from functools import partial
from pathlib import Path
from argparse import Namespace

import torch

from vidlu import defaults, models, parameters
from vidlu.data_utils import CachingDatasetFactory
from vidlu.data import DataLoader
from vidlu.training import Trainer
from vidlu.utils import tree
from vidlu.utils.collections import NameDict
from vidlu.utils.func import (argtree_partial, find_empty_params_deep,
                              ArgTree, params, Empty, default_args, EscapedArgTree)

import vidlu.data_utils.dataset_ops as do
import vidlu.utils.tree as vut


# Factory messages #################################################################################

def _print_all_args_message(func):
    print("All arguments:")
    print(f"Argument tree ({func.func}):")
    tree.print_tree(ArgTree.from_func(func), depth=1)


def _print_missing_args_message(func):
    empty_args = list(find_empty_params_deep(func))
    if len(empty_args) != 0:
        print("Unassigned arguments:")
        for ea in empty_args:
            print(f"  {'/'.join(ea)}")


def _print_args_messages(kind, type_, factory, argtree, verbosity=1):
    if verbosity > 0:
        print(f'{kind}:', type_.__name__)
        tree.print_tree(argtree, depth=1)
        if verbosity > 1:
            _print_all_args_message(factory)
        _print_missing_args_message(factory)


# Data #############################################################################################

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


def get_data(data_str: str, datasets_dir, cache_dir=None) -> dict:
    """

    Args:
        data_str (str): a string representing the datasets.
        datasets_dir: directory with
        cache_dir:

    Returns:

    """
    from vidlu import data

    if ':' in data_str:
        data_str, transform_str = data_str.split(':', 1)
    else:
        transform_str = None

    name_options_subsets_tuples = parse_data_str(data_str)

    get_parted_dataset = data.DatasetFactory(datasets_dir)
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

    if transform_str is not None:
        flat_data = dict([((k1, k2), v) for k1, d in data.items() for k2, v in d.items()])
        values = eval(transform_str, dict(d=list(flat_data.values()),
                                          **{k: v for k, v in vars(do).items()
                                             if not k.startswith('_')}))
        data = dict(((f'data{i}', {'sub0': v}) for i, v in enumerate(values)))
    return data


get_data.help = \
    ('Dataset configuration with syntax'
     + ' "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)",'
     + ' e.g. "cifar10{train,val}", "cityscapes(remove_hood=True){trainval,val,test}",'
     + ' "inaturalist{train,all}", or "camvid{trainval}, wilddash(downsampling=2){val}"')


def get_data_preparation(*datasets):
    dataset = datasets[0]
    from vidlu.transforms.input_preparation import prepare_input_image, prepare_label
    fields = tuple(dataset[0].keys())
    if set('xy').issubset(fields):
        return lambda ds: ds.map_fields(dict(x=prepare_input_image, y=prepare_label),
                                        func_name='prepare')
    raise ValueError(f"Unknown record format: {fields}.")


def get_prepared_data(data_str: str, datasets_dir, cache_dir):
    data = get_data(data_str, datasets_dir, cache_dir)
    datasets = dict(tree.flatten(data)).values()
    prepare = get_data_preparation(*datasets)
    return tuple(map(prepare, datasets))


def get_prepared_data_for_trainer(data_str: str, datasets_dir, cache_dir):
    if ':' in data_str:
        names, data_str = [x.strip() for x in data_str.split(':', 1)]
        names = [x.strip() for x in names.split(',')]
        if not all(x.startswith('train') or x.startswith('test') for x in names):
            raise ValueError('All dataset identifiers should start with either "train" or "test".'
                             + f' Some of {names} do not.')
    else:
        names = ['train', 'test']
    data = get_data(data_str, datasets_dir, cache_dir)
    datasets = dict(tree.flatten(data)).values()
    prepare = get_data_preparation(*datasets)
    datasets = map(prepare, datasets)
    names_iter = iter(names)
    print("Datasets: " + ", ".join(f"{name}.{k}({len(ds)}) as {next(names_iter)}"
                                   for name, subsets in data.items()
                                   for k, ds in subsets.items()))
    return NameDict(**dict(zip(names, datasets)))


# Model ############################################################################################

def get_input_adapter(input_adapter_str, *, problem, data_statistics=None):
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
    import vidlu.modules as M
    from vidlu.problem import Supervised
    from vidlu.transforms import image as imt
    if isinstance(problem, Supervised):
        if input_adapter_str.startswith("standardize"):
            if input_adapter_str == "standardize":
                stats = dict(mean=torch.from_numpy(data_statistics.mean),
                             std=torch.from_numpy(data_statistics.std))
            else:
                stats = eval("dict(" + input_adapter_str[len("standardize("):])
                stats = {k: torch.tensor(v) for k, v in stats.items()}
            return M.Func(imt.Standardize(**stats), imt.Destandardize(**stats))
        elif input_adapter_str == "id":  # min 0, max 1 is expected for images
            return M.Identity()
        else:
            try:
                return eval(input_adapter_str)
            except Exception as e:
                raise ValueError(f"Invalid input_adapter_str: {input_adapter_str}, \n{e}")
    raise NotImplementedError()


def get_model(model_str: str, *, input_adapter_str='id', problem=None, init_input=None,
              prep_dataset=None, device=None, verbosity=1) -> torch.nn.Module:
    from torch import nn
    import vidlu.modules as vm
    import vidlu.modules.components as vmc
    import torchvision.models as tvmodels

    if prep_dataset is None and (problem is None or init_input is None):
        raise ValueError(
            "If `prep_dataset` is `None`, `problem` and `init_input` need to be provided.")

    if problem is None:
        problem = defaults.get_problem_from_dataset(prep_dataset)
    if init_input is None and prep_dataset is not None:
        init_input = next(iter(DataLoader(prep_dataset, batch_size=1)))[0]

    # `argtree_arg` has at most 1 element because `maxsplit`=1
    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    model_class = getattr(models, model_name) if hasattr(models, model_name) else eval(model_name)
    argtree = defaults.get_model_argtree(model_class, problem)
    argtree_arg = (
        eval(f"t({argtree_arg[0]})",
             dict(nn=nn, vm=vm, vmc=vmc, models=models, tvmodels=tvmodels, t=ArgTree,
                  partial=partial))
        if len(argtree_arg) == 1 else ArgTree())
    argtree.update(argtree_arg)

    model_f = argtree_partial(
        model_class,
        **argtree,
        input_adapter=get_input_adapter(
            input_adapter_str,
            problem=problem,
            data_statistics=(None if prep_dataset is None
                             else prep_dataset.info.cache['standardization'])))

    _print_args_messages('Model', model_class, model_f, argtree, verbosity=verbosity)

    model = model_f()
    model.eval()

    if device is not None:
        model.to(device)
        init_input = init_input.to(device)
    if hasattr(model, 'initialize'):
        model.initialize(init_input)
    else:
        warnings.warn("The model does not have an initialize method.")
        model(init_input)
    if device is not None:
        model.to(device)

    if verbosity > 1:
        print(model)
    print('Parameter count:', vm.parameter_count(model))

    return model


get_model.help = \
    ('Model configuration with syntax "Model[,arg1=value1, arg2=value2, ..]",'
     + ' where values can be ArgTrees. '
     + 'Instead of ArgTree(...), t(...) can be used. '
     + 'Example: "ResNet,base_f=a(depth=34, base_width=64, small_input=True, _f=a(dropout=True)))"')

"configs.supervised,configs.classification,configs.resnet_cifar"


# Initial pre-trained parameters ###################################################################

def _parse_parameter_translation_string(params_str):
    module_re = fr'\w+(?:\.\w+)*'
    regex = re.compile(
        fr'(?P<translator>[\w]+)(?:\[(?P<src_dict>{module_re})\])?'
        + fr'(?:\((?P<src_module>{module_re})\))?'
        + fr'(?:,(?P<dest_module>{module_re}))?(?::(?P<file>.+))?')
    m = regex.fullmatch(params_str.strip())
    if m is None:
        raise ValueError('`params_str` does not match the pattern'
                         + ' "translator[[<src_dict>]][(<src_module>)][,<dest_module>][!][:file]".')
    return Namespace(
        **{k: m.group(k) or '' for k in
           ['translator', 'src_dict', 'src_module', 'dest_module', 'file']})


def get_translated_parameters(params_str, *, params_dir=None, state_dict=None):
    """Loads pretrained parameters into a model (or its submodule) from a file.

    Args:
        params_str (str): a string in the format
            "translator[(src_module)][,dest_module][:file]", where "translator"
            represents the function for loading and translating parameter names,
            "src_module" (optional) is for extracting a part of parameters (with
            translated names) with a common submodule and represents the
            submodule name, "dest_module" (optional) represents the name of the
            submodule that the parameters are to be loaded in, "file" represents
            the file name (with extension) that the parameters are to be loaded
            from. The "src_module" substring is is removed from the beginning of
            translated parameter names and the "dest_module" is added added
            instead.
        state_dict: a dictionary with parameters. It should be provided if there
            is no file path at the end of `params_str`.
        params_dir: the directory that the file with parameters is in.
    """
    p = _parse_parameter_translation_string(params_str)
    # load the state and filter and remove `src_module` from module names
    if not ((state_dict is None) ^ (p.file == '')):
        raise RuntimeError('Either state_dict should be provided or params_str should contain the'
                           + ' parameters file path at the end of `params_str`.')
    if p.file != '':
        path = Path(p.file)
        if not path.is_absolute():
            path = Path(params_dir) / p.file
        state_dict = parameters.get_translated_parameters(p.translator, path, subdict=p.src_dict)
    else:
        state_dict = parameters.get_translated_parameters(p.translator, state_dict,
                                                          subdict=p.src_dict)
    if len(p.src_module) > 0:  # or len(p.dest_module) > 0:
        # start = f'{p.dest_module}.' if len(p.dest_module) > 0 else ''
        # state_dict = {start + k[len(p.src_module) + 1:]: v for k, v in state_dict.items()
        #              if k.startswith(p.src_module)}
        state_dict = {k[len(p.src_module) + 1:]: v for k, v in state_dict.items()
                      if k.startswith(p.src_module)}
    return state_dict, p.dest_module


# Training and evaluation ##########################################################################

# noinspection PyUnresolvedReferences
def get_trainer(trainer_str: str, *, dataset, model, verbosity=1) -> Trainer:
    import math
    from torch import optim
    from torch.optim import lr_scheduler
    from vidlu.modules import losses
    import vidlu.training.adversarial as ta
    import vidlu.training.configs as tc
    import vidlu.training.extensions as te
    import vidlu.training.steps as ts
    from vidlu.training.adversarial import attacks
    from vidlu.transforms import jitter
    t = ArgTree

    config = eval(f"tc.TrainerConfig({trainer_str})")

    default_config = tc.TrainerConfig(**defaults.get_trainer_args(config.extension_fs, dataset))
    if 'optimizer_f' in config and 'weight_decay' in default_args(config.optimizer_f):
        raise RuntimeError("The `weight_decay` argument should be passed to the trainer directly"
                           + " instead of to the optimizer.")
    trainer_f = partial(Trainer, **tc.to_trainer_args(default_config, config))

    _print_args_messages('Trainer', Trainer, factory=trainer_f, argtree=config, verbosity=verbosity)

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

    return default_metrics + additional_metrics
