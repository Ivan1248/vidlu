import re
from argparse import Namespace
import typing as T

import torch

from vidlu import models, metrics
import vidlu.modules as vm
from vidlu.models import params as mparams
import vidlu.data.utils as vdu
from vidlu.data import DataLoader, Dataset, Record
from vidlu.training import Trainer
from vidlu.utils import tree
from vidlu.utils.collections import NameDict
import vidlu.utils.func as vuf
from vidlu.utils.func import Reserved
from vidlu.utils.func import partial
from vidlu.extensions import extensions
from vidlu.transforms.data_preparation import prepare_element

from . import defaults


# eval

def factory_eval(expr: str, globals=None, locals=None):
    try:
        return eval(expr, globals, locals)
    except NameError as e:
        available_names = sorted([*(globals or {}).keys(), *(locals or {}).keys()])
        raise NameError(f"{e.args[0]}. Available: {str(available_names)[1:-1]}.", *e.args[1:])


def module_to_dict(module):
    return {k: v for k, v in vars(module).items() if not k.startswith("_")}


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


def apply_default_transforms(datasets, cache_dir):
    # TODO: de-hardcode
    default_transforms = [  # partial(vdu.add_pixel_stats_to_info_lazily, cache_dir=cache_dir),
        partial(vdu.add_segmentation_class_info_lazily, cache_dir=cache_dir),
        partial(vdu.cache_lazily, cache_dir=cache_dir)]
    for i, ds in enumerate(datasets):
        for transform in default_transforms:
            ds = transform(ds)
        datasets[i] = ds
    return datasets


def get_data_namespace():
    import vidlu.transforms as vt
    import torchvision.transforms.functional_tensor as tvt
    import vidlu.modules.functional as vmf
    from vidlu.data.datasets import taxonomies

    namespace = {**module_to_dict(tvt), **module_to_dict(vmf), **module_to_dict(vt),
                 **module_to_dict(vdu.dataset_ops)}
    namespace.update(vt=vt, taxonomies=taxonomies, Record=Record)
    return namespace


def evaluate_data_transforms_string(datasets, transform_str):
    # TODO: improve names if necessary
    return factory_eval(transform_str, dict(**get_data_namespace(), d=datasets))


def get_data(data_str: str, datasets_dir, cache_dir=None, return_datasets_only=False,
             factory_version=1) \
        -> T.Sequence[T.Tuple[T.Tuple[str], Dataset]]:
    if factory_version > 1:
        return get_data_new(data_str, datasets_dir, cache_dir)

    from vidlu import data
    from vidlu.data import Record

    if ':' in data_str:
        data_str, transform_str = data_str.split(':', 1)
    else:
        transform_str = None

    name_options_subsets_tuples = parse_data_str(data_str)

    get_dataset = data.DatasetFactory(datasets_dir)

    keys = []
    datasets = []
    for name, options_str, subsets in name_options_subsets_tuples:
        options = factory_eval(f'dict{options_str or "()"}', dict(Record=Record))
        full_name = f"{name}{options_str or ''}"
        for subset in subsets:
            ds = get_dataset(name, subset=subset, **options)
            keys.append((full_name, subset))
            datasets.append(ds)

    datasets = apply_default_transforms(datasets, cache_dir=cache_dir)

    if transform_str is not None:  # TODO: improve names if necessary
        datasets = evaluate_data_transforms_string(datasets, transform_str)

    if return_datasets_only:
        return datasets
    return datasets, keys, transform_str


get_data.help = \
    ('Dataset configuration with syntax'
     + ' "(dataset1`{`(subset1_1, ..)`}`, dataset2`{`(subset2_1, ..)`}`, ..)",'
     + ' e.g. "cifar10{train,val}", "cityscapes(remove_hood=True){trainval,val,test}",'
     + ' "inaturalist{train,all}", or "camvid{trainval}, wilddash(downsampling=2){val}"')


def get_default_transforms(cache_dir):
    return dict(
        add_pixel_stats=partial(vdu.add_pixel_stats_to_info_lazily, cache_dir=cache_dir),
        add_seg_class_info=partial(vdu.add_segmentation_class_info_lazily, cache_dir=cache_dir),
        cache=partial(vdu.cache_lazily, cache_dir=cache_dir))


def get_data_new(data_str: str, datasets_dir, cache_dir=None) \
        -> T.Sequence[T.Tuple[T.Tuple[str], Dataset]]:
    from vidlu import data

    factories = data.DatasetFactory(datasets_dir).as_namespace()
    dt = NameDict(get_default_transforms(cache_dir))
    glob = {**get_data_namespace(), **factories.__dict__, **dt, 'extend': lambda ds: dt.cache(
        dt.add_seg_class_info(ds))}

    try:
        data = factory_eval(data_str, glob)
    except SyntaxError as e:
        # loc = {}
        exec(data_str, glob)
        data = glob['data']
    return data


def prepare_dataset(dataset):
    return dataset.map(lambda r: Record({k: prepare_element(v, k) for k, v in r.items()}),
                       func_name='prepare')


def get_prepared_data_for_trainer(data_str: str, datasets_dir, cache_dir, factory_version=1):
    names = None
    if factory_version == 1 and ':' in data_str:  # TODO: remove names
        names, data_str = [x.strip() for x in data_str.split(':', 1)]
        names = [x.strip() for x in names.split(',')]
        if not all(x.startswith('train') or x.startswith('test') for x in names):
            raise ValueError(
                'All dataset identifiers should start with either "train" or "test".'
                + f' Some of {names} do not.')

    datasets = get_data(data_str, datasets_dir, cache_dir, return_datasets_only=True,
                        factory_version=factory_version)
    if not isinstance(datasets, T.Mapping):
        if names is None:
            names = [f'train{i}' for i in range(len(datasets) - 1)] if len(datasets) > 2 else [
                'train']
            names.extend([f'test{i}' for i in range(len(datasets[0]) - 1)] if isinstance(
                datasets[-1], tuple) else ['test'])
            datasets = list(datasets[:-1]) + (list(datasets[-1]) if isinstance(
                datasets[-1], tuple) else [datasets[-1]])
        datasets = dict(zip(names, datasets))

    datasets = {k: prepare_dataset(ds) for k, ds in datasets.items()}
    print("Datasets:\n" + "  \n ".join(f"{name}: {getattr(ds, 'identifier', ds)}), size={len(ds)}"
                                       for name, ds in datasets.items()))
    return NameDict(datasets)


# Model ############################################################################################

def get_input_adapter(input_adapter_str, *, data_stats=None):
    """Returns a bijective module to be inserted before the model to scale
    inputs.

    For images, it is assumed that the input elements are in range 0 to 1.

    Args:
        input_adapter_str: a string in {"standardize", "id"}
        problem (ProblemInfo):
        data_stats (optional): a namespace containing the mean and the standard
            deviation ("std") of the training dataset.

    Returns:
        A torch module.
    """
    import vidlu.modules as M
    from vidlu.transforms import image as imt
    import vidlu.configs.data.stats as cds

    if input_adapter_str.startswith("standardize"):
        if input_adapter_str == "standardize":
            stats = dict(mean=torch.from_numpy(data_stats.mean),
                         std=torch.from_numpy(data_stats.std))
        else:
            stats = factory_eval("dict(" + input_adapter_str[len("standardize("):],
                                 {**vars(cds), **extensions})
            stats = {k: torch.tensor(v) for k, v in stats.items()}
        return M.Func(imt.Standardize(**stats), imt.Destandardize(**stats))
    elif input_adapter_str == "id":  # min 0, max 1 is expected for images
        return M.Identity()
    else:
        try:
            return factory_eval(input_adapter_str)
        except Exception as e:
            raise ValueError(f"Invalid input_adapter_str: {input_adapter_str}, \n{e}")
    raise NotImplementedError()


def build_and_init_model(model, init_input, device):
    model.eval()
    if device is not None:
        model.to(device)
        init_input = init_input.to(device)
    if min(init_input.shape[-2:]) > 128:  # smaller input for faster initialization
        init_input = init_input[:, :, :128, :128]
    if hasattr(model, 'initialize'):
        model.initialize(init_input)
    else:
        vm.call_if_not_built(model, init_input)


_func_short = dict(partial=partial, t=vuf.ArgTree, ft=vuf.FuncTree, ot=vuf.ObjectUpdatree,
                   sot=vuf.StrictObjectUpdatree, it=vuf.IndexableUpdatree,
                   sit=vuf.StrictIndexableUpdatree, torch=torch)


def get_model(model_str: str, *, input_adapter_str='id', problem=None, init_input=None,
              prep_dataset=None, device=None, verbosity=1) -> torch.nn.Module:
    from torch import nn
    import vidlu.modules as vm
    import vidlu.modules.components as vmc
    import vidlu.modules.other as vmo
    import torchvision.models as tvmodels
    from fractions import Fraction as Frac

    namespace = dict(nn=nn, vm=vm, vmc=vmc, vmo=vmo, models=models, tvmodels=tvmodels,
                     Reserved=Reserved, Frac=Frac, **_func_short, **extensions)

    if prep_dataset is None:
        if problem is None or init_input is None:
            raise ValueError("`problem` and `init_input` are required if `prep_dataset` is `None`.")
    else:
        problem = problem or defaults.get_problem_from_dataset(prep_dataset)

    if init_input is None and prep_dataset is not None:
        init_input = next(iter(DataLoader([prep_dataset[0]] * 2, batch_size=2)))[0]

    # `argtree_arg` has at most 1 element because `maxsplit`=1
    model_name, *argtree_arg = (x.strip() for x in model_str.strip().split(',', 1))

    if model_name[0] in "'\"":  # torch.hub
        print(input_adapter_str)
        assert input_adapter_str == 'id'
        model = factory_eval(f"torch.hub.load({model_str})")
    else:
        if hasattr(models, model_name):
            model_f = getattr(models, model_name)
        else:
            model_f = factory_eval(model_name, namespace)
        model_class = model_f

        argtree = defaults.get_model_argtree_for_problem(model_f, problem)
        if len(argtree_arg) != 0:
            argtree.update(factory_eval(f"t({argtree_arg[0]})", namespace))
        model_f = argtree.apply(model_f)
        input_adapter = get_input_adapter(
            input_adapter_str, data_stats=prep_dataset.info[
                'pixel_stats'] if input_adapter_str == 'standardize' else None)
        _print_args_messages('Model', model_class, model_f,
                             {**argtree, 'input_adapter': input_adapter},
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
    src_re = fr'([\w\/]*)?{dst_re}'
    regex = re.compile(
        fr'(?P<transl>\w+)(?::(?P<src>{src_re})?(?:->(?P<dst>{dst_re}))?)?:(?P<name>.+)')
    m = regex.fullmatch(params_str.strip())
    if m is None:
        regex = re.compile(
            fr'(?P<transl>\w+)(?:\[(?P<src>{src_re})])?(?:->(?P<dst>{dst_re}))?(?::(?P<name>.+))?')
        m = regex.fullmatch(params_str.strip())
    if m is None:
        # raise ValueError(
        #     f'{params_str=} does not match the pattern "translator[[<src>]][-><dst>][:<name>]".'
        #     + " <src> supports indexing nested dictionaries by putting commas between keys.")
        raise ValueError(
            f'{params_str=} does not match the pattern "<translator>[[<src>]][-><dst>]:<name>".'
            + ' <src> supports indexing nested dictionaries by putting "/" between keys.')
    p1 = Namespace(**{k: m.group(k) or '' for k in ['transl', 'src', 'dst', 'name']})
    *src_dict, src = p1.src.split("/")
    return Namespace(translator=p1.transl, src_dict=src_dict, src_module=src, dest_module=p1.dst,
                     name=p1.name)


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
    if not ((state_dict is None) ^ (p.name == '')):
        raise RuntimeError('Either state_dict should be provided or params_str should contain the'
                           + ' parameters file path at the end of `params_str`.')
    if p.name != '':
        path = mparams.get_path(p.name, params_dir)
        state_dict = torch.load(path)
    state_dict = mparams.get_translated_parameters(p.translator, state_dict, subdict=p.src_dict)
    state_dict_fr = mparams.filter_by_and_remove_key_prefix(state_dict, p.src_module,
                                                            error_on_no_match=True)
    return state_dict_fr, p.dest_module


# Training and evaluation ##########################################################################

# noinspection PyUnresolvedReferences
def short_symbols_for_get_trainer():
    import math
    import os
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
    from vidlu.data import class_mapping
    tc = ct  # backward compatibility
    return {**locals(), **_func_short, **extensions}


def get_trainer(trainer_str: str, *, dataset, model, deterministic=False,
                verbosity=1) -> Trainer:
    import vidlu.configs.training as ct

    default_config = ct.TrainerConfig(**defaults.get_trainer_args(dataset))  # empty
    ah = factory_eval(f"vuf.ArgHolder({trainer_str})", short_symbols_for_get_trainer())
    config = ct.TrainerConfig(default_config, *ah.args)
    updatree = vuf.ObjectUpdatree(**ah.kwargs)
    config = updatree.apply(config)

    trainer_f = partial(Trainer, **config.normalized())

    trainer = trainer_f(model=model, deterministic=deterministic)
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
