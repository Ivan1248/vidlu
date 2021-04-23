import datetime
import os
import re
from pathlib import Path
from argparse import Namespace
import typing as T
from warnings import warn

import torch
import numpy as np

from vidlu import models, metrics
import vidlu.modules as vm
from vidlu.models import params
import vidlu.data.utils as vdu
from vidlu.data import DataLoader, Dataset
from vidlu.training import Trainer
from vidlu.utils import tree
from vidlu.utils.collections import NameDict
import vidlu.utils.func as vuf
from vidlu.utils.func import Reserved
from vidlu.utils.func import partial
from vidlu.utils.misc import import_module

from . import defaults

# eval
unsafe_eval = eval


def module_to_dict(module):
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


# Extensions #######################################################################################

def get_extensions():
    paths = os.environ.get("VIDLU_EXTENSIONS", None)
    if paths is None:
        return []
    paths = [Path(p) for p in paths.split(os.path.sep)]  # split on ':' (Linux)
    for p in paths:
        if not p.exists():
            warn(f"Extension path {p} does not exist.")
    return map(import_module, paths)


extensions = {e.__name__: e for e in get_extensions()}


# Factory messages #################################################################################

def _print_all_args_message(func):
    print("All arguments:")
    print(f"Argument tree ({func.func if isinstance(func, partial) else func}):")
    tree.print_tree(vuf.ArgTree.from_func(func), depth=1)


def _print_missing_args_message(func):
    empty_args = list(vuf.find_params_deep(func, lambda k, v: vuf.is_empty(v)))
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
    single_ds_re = re.compile(fr'({start_re}{name_re}{options_re}?{subsets_re}?)')
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


def get_data(data_str: str, datasets_dir, cache_dir=None) \
        -> T.Sequence[T.Tuple[T.Tuple[str], Dataset]]:
    """

    Args:
        data_str (str): a string representing the datasets.
        datasets_dir: directory with
        cache_dir:

    Returns:

    """
    from vidlu import data
    from vidlu.data import Record
    import vidlu.transforms as vt
    import torchvision.transforms.functional_tensor as tvt
    import vidlu.modules.functional as vmf

    if ':' in data_str:
        data_str, transform_str = data_str.split(':', 1)
    else:
        transform_str = None

    name_options_subsets_tuples = parse_data_str(data_str)

    get_parted_dataset = data.DatasetFactory(datasets_dir)
    if cache_dir is not None:
        add_stats = partial(vdu.add_image_statistics_to_info_lazily, cache_dir=cache_dir)
        get_parted_dataset = vdu.CachingDatasetFactory(get_parted_dataset, cache_dir, [add_stats])

    data = []
    for name, options_str, subsets in name_options_subsets_tuples:
        options = unsafe_eval(f'dict{options_str or "()"}', dict(Record=Record))
        pds = get_parted_dataset(name, **options)
        k = f"{name}{options_str or ''}"
        for s in subsets:
            data.append(((k, s), getattr(pds, s)))

    if transform_str is not None:  # TODO: improve names if necessary
        namespace = {**module_to_dict(tvt), **module_to_dict(vmf), **module_to_dict(vt),
                     **module_to_dict(vdu.dataset_ops)}
        namespace.update(vt=vt, Record=Record, d=[v for k, v in data])
        values = unsafe_eval(transform_str, namespace)
        data = [((getattr(v, 'identifier', f'dataset{i}'), ''), v) for i, v in enumerate(values)]
    return data  # not dict since elements can repeat


get_data.help = \
    ('Dataset configuration with syntax'
     + ' "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)",'
     + ' e.g. "cifar10{train,val}", "cityscapes(remove_hood=True){trainval,val,test}",'
     + ' "inaturalist{train,all}", or "camvid{trainval}, wilddash(downsampling=2){val}"')


def get_data_preparation(dataset):
    from vidlu.transforms.input_preparation import prepare_input_image, prepare_label
    fields = tuple(dataset[0].keys())
    if set('xy').issubset(fields):
        return lambda ds: ds.map_fields(dict(x=prepare_input_image, y=prepare_label),
                                        func_name='prepare')
    elif fields == ('x',):
        return lambda ds: ds.map_fields(dict(x=prepare_input_image), func_name='prepare_x')
    raise ValueError(f"Unknown record format: {fields}.")


def prepare_data(data):
    datasets = [ds for _, ds in data]
    preparers = [get_data_preparation(ds) for ds in datasets]
    return tuple(prepare(ds) for prepare, ds in zip(preparers, datasets))


def get_prepared_data(data_str: str, datasets_dir, cache_dir):
    return prepare_data(get_data(data_str, datasets_dir, cache_dir))


def get_prepared_data_for_trainer(data_str: str, datasets_dir, cache_dir):
    if ':' in data_str:  # TODO: remove names
        names, data_str = [x.strip() for x in data_str.split(':', 1)]
        names = [x.strip() for x in names.split(',')]
        if not all(x.startswith('train') or x.startswith('test') for x in names):
            raise ValueError('All dataset identifiers should start with either "train" or "test".'
                             + f' Some of {names} do not.')
    else:
        names = ['train', 'test']

    datasets = get_prepared_data(data_str, datasets_dir, cache_dir)
    named_datasets = NameDict(dict(zip(names, datasets)))
    print("Datasets:\n" + "  \n ".join(f"{name}: {getattr(ds, 'identifier', ds)}), size={len(ds)}"
                                       for name, ds in named_datasets.items()))
    return named_datasets


# Model ############################################################################################

def get_input_adapter(input_adapter_str, *, problem, data_statistics=None):
    """Returns a bijective module to be inserted before the model to scale
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
    from vidlu.factories.problem import Supervised
    from vidlu.transforms import image as imt
    import vidlu.configs.data.stats as cds
    if isinstance(problem, Supervised):
        if input_adapter_str.startswith("standardize"):
            if input_adapter_str == "standardize":
                stats = dict(mean=torch.from_numpy(data_statistics.mean),
                             std=torch.from_numpy(data_statistics.std))
            else:
                stats = unsafe_eval("dict(" + input_adapter_str[len("standardize("):],
                                    {**vars(cds), **extensions})
                stats = {k: torch.tensor(v) for k, v in stats.items()}
            return M.Func(imt.Standardize(**stats), imt.Destandardize(**stats))
        elif input_adapter_str == "id":  # min 0, max 1 is expected for images
            return M.Identity()
        else:
            try:
                return unsafe_eval(input_adapter_str)
            except Exception as e:
                raise ValueError(f"Invalid input_adapter_str: {input_adapter_str}, \n{e}")
    raise NotImplementedError()


def build_and_init_model(model, init_input, device):
    if device is not None:
        model.to(device)
        init_input = init_input.to(device)
    if np.array(init_input.shape[-2:]).min() > 128:  # smaller input for faster initialization
        init_input = init_input[:, :, :128, :128]
    if hasattr(model, 'initialize'):
        model.initialize(init_input)
    else:
        vm.call_if_not_built(model, init_input)


def get_model(model_str: str, *, input_adapter_str='id', problem=None, init_input=None,
              prep_dataset=None, device=None, verbosity=1) -> torch.nn.Module:
    from torch import nn
    import vidlu.modules as vm
    import vidlu.modules.components as vmc
    import vidlu.modules.other as vmo
    import torchvision.models as tvmodels
    from fractions import Fraction as Frac

    namespace = dict(nn=nn, vm=vm, vmc=vmc, vmo=vmo, models=models, tvmodels=tvmodels,
                     t=vuf.ArgTree,
                     partial=partial, Reserved=Reserved, Frac=Frac, **extensions)

    if prep_dataset is None:
        if problem is None or init_input is None:
            raise ValueError("`problem` and `init_input` are required if `prep_dataset` is `None`.")
    else:
        problem = problem or defaults.get_problem_from_dataset(prep_dataset)

    if init_input is None and prep_dataset is not None:
        init_input = next(iter(DataLoader(prep_dataset, batch_size=2)))[0]

    # `argtree_arg` has at most 1 element because `maxsplit`=1
    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    if model_name[0] in "'\"":  # torch.hub
        return unsafe_eval(f"torch.hub.load({model_str})")
    elif hasattr(models, model_name):
        model_f = getattr(models, model_name)
    else:
        for ext in extensions.values():
            if model_name in vars(ext):
                model_f = getattr(ext, model_name)
                break
        else:
            model_f = unsafe_eval(model_name, namespace)
    model_class = model_f

    argtree = defaults.get_model_argtree_for_problem(model_f, problem)
    input_adapter = get_input_adapter(
        input_adapter_str, problem=problem,
        data_statistics=(None if prep_dataset is None
                         else prep_dataset.info.cache['standardization']))
    if len(argtree_arg) != 0:
        argtree.update(unsafe_eval(f"t({argtree_arg[0]})", namespace))
    if len(argtree) > 0:
        model_f = vuf.argtree_partial(model_f, **argtree)
    _print_args_messages('Model', model_class, model_f, {**argtree, 'input_adapter': input_adapter},
                         verbosity=verbosity)
    if "input_adapter" in vuf.params(model_f):
        model = model_f(input_adapter=input_adapter)
    else:
        model = model_f()
        if input_adapter_str != 'id':
            model.register_forward_pre_hook(lambda m, x: input_adapter(*x))

    build_and_init_model(model, init_input, device)
    model.eval()

    if verbosity > 2:
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
    dst_re = fr'\w+(?:\.\w+)*'
    src_re = fr'(\w+(?:,\w+)*)?{dst_re}'
    regex = re.compile(
        fr'(?P<transl>[\w]+)(?:\[(?P<src>{src_re})])?(?:->(?P<dst>{dst_re}))?(?::(?P<file>.+))?')
    m = regex.fullmatch(params_str.strip())
    if m is None:
        raise ValueError(
            f'{params_str=} does not match the pattern "translator[[<src>]][-><dst>][!][:file]".'
            + " <src> supports indexing nested dictionaries by putting commas between keys.")
    p1 = Namespace(**{k: m.group(k) or '' for k in ['transl', 'src', 'dst', 'file']})
    *src_dict, src = p1.src.split(",")
    return Namespace(translator=p1.transl, src_dict=src_dict, src_module=src, dest_module=p1.dst,
                     file=p1.file)


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
        state_dict = params.load_params_file(path if (path := Path(p.file)).is_absolute() else
                                             Path(params_dir) / path)
    state_dict = params.get_translated_parameters(p.translator, state_dict, subdict=p.src_dict)
    state_dict_fr = params.filter_by_and_remove_key_prefix(state_dict, p.src_module,
                                                           error_on_no_match=True)
    return state_dict_fr, p.dest_module


# Training and evaluation ##########################################################################

# noinspection PyUnresolvedReferences
def short_symbols_for_get_trainer():
    import math
    from torch import optim
    import vidlu.optim.lr_schedulers as lr
    from vidlu.modules import losses
    import vidlu.data as vd
    import vidlu.training.robustness as ta
    import vidlu.configs.training as ct
    import vidlu.configs.robustness as cr
    import vidlu.training.steps as ts
    from vidlu.training.robustness import attacks
    from vidlu.transforms import jitter
    import vidlu.utils.func as vuf
    from vidlu.utils.func import partial
    tc = ct  # backward compatibility
    t = vuf.ArgTree
    return locals()


def get_trainer(trainer_str: str, *, dataset, model, verbosity=1) -> Trainer:
    import vidlu.configs.training as ct

    ah = unsafe_eval(f"vuf.ArgHolder({trainer_str})", short_symbols_for_get_trainer())
    config = ct.TrainerConfig(*ah.args, **ah.kwargs)

    default_config = ct.TrainerConfig(**defaults.get_trainer_args(config.extension_fs, dataset))
    trainer_f = partial(Trainer, **ct.TrainerConfig(default_config, config).normalized())

    trainer = trainer_f(model=model)
    _print_args_messages('Trainer', Trainer, factory=trainer_f, argtree=config, verbosity=verbosity)
    return trainer


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
    if problem is None:
        if dataset is None:
            raise ValueError("get_metrics: either the dataset argument"
                             + " or the problem argument need to be given.")
        problem = defaults.get_problem_from_dataset(dataset)

    default_metrics, main_metrics = defaults.get_metrics(trainer, problem)

    metrics_str = metrics_str.strip()
    metric_names = [x.strip() for x in metrics_str.strip().split(',')] if metrics_str else []
    additional_metrics = [getattr(metrics, name) for name in metric_names]

    return default_metrics + additional_metrics, main_metrics
