import numpy as np
from tensorflow.python import pywrap_tensorflow
import re


def get_parameters_from_checkpoint_file(path, param_name_filter=lambda x: True):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return {
        key: reader.get_tensor(key)
        for key in sorted(var_to_shape_map) if param_name_filter(key)
    }


def translate_variable_names(names: list, simple_replacements,
                             generic_replacements):

    def replace_simple(string, translations):
        for si, so in translations:
            string = string.replace(si, so)
        return string

    def replace_generic(string, translations):
        for pin, (pout, f) in translations:

            def replace(match):
                g1 = match.group(1)
                return replace_simple(pout, [('{}', f(g1))])

            string = re.sub(pin, replace, string)
        return string

    names = map(lambda s: replace_simple(s, simple_replacements), names)
    names = map(lambda s: replace_generic(s, generic_replacements), names)
    return list(names)


def get_resnet_parameters_from_checkpoint_file(checkpoint_file_path, include_first_conv=True):
    # Checkpoint files can be downloaded here:
    # https://github.com/tensorflow/models/tree/master/research/slim
    drop_out_words = ['oving', 'Momentum', 'global_step', 'logits', 'bias']
    if not include_first_conv:
        drop_out_words += ['resnet_v2_50/conv1']
    filters = [lambda x: not any(map(lambda d: d in x, drop_out_words))]

    def filter_param_name(param_name):
        return all(map(lambda f: f(param_name), filters))

    param_name_to_array = get_parameters_from_checkpoint_file(
        checkpoint_file_path, filter_param_name)

    simple_replacements = [  # by priority
        ('resnet_v2_50/conv1', 'resnet_root_block/conv'),
        ('resnet_v2_50/logits', 'conv_logits'),
        ('resnet_v2_50', 'resnet_middle'),
        ('unit_1/bottleneck_v2/preact', 'rb0/bn_relu/bn'),
        ('bottleneck_v2/preact', 'block/bn_relu0/bn'),
        ('bottleneck_v2/shortcut', 'conv_skip'),
        ('postnorm', 'post_bn_relu/bn'),
        ('weights', 'weights:0'),
        ('biases', 'bias:0'),
        ('beta', 'offset:0'),
        ('gamma', 'scale:0'),
    ]
    id = lambda x: x
    sub1 = lambda x: str(int(x) - 1)
    generic_replacements = [
        (r'block(\d+)', ['group{}', sub1]),
        (r'unit_(\d+)', ['rb{}', sub1]),
        (r'bottleneck_v2\/conv(\d+)\/BatchNorm', ['block/bn_relu{}/bn', id]),
        (r'bottleneck_v2\/conv(\d+)', ['block/conv{}', sub1]),
    ]

    old_names = param_name_to_array.keys()
    new_names = translate_variable_names(old_names, simple_replacements,
                                         generic_replacements)
    old_name_to_new_name = dict(zip(old_names, new_names))
    return {
        nn: param_name_to_array[on]
        for on, nn in old_name_to_new_name.items()
    }


def get_densenet_parameters_from_checkpoint_file(checkpoint_file_path):
    # Checkpoint files can be downloaded here:
    # https://github.com/pudae/tensorflow-densenet
    drop_out_words = [
        'oving', 'Momentum', 'global_step', 'logits', 'bias',
        'densenet121/BatchNorm'
    ]
    filters = [lambda x: not any(map(lambda d: d in x, drop_out_words))]

    def filter_param_name(param_name):
        return all(map(lambda f: f(param_name), filters))

    param_name_to_array = get_parameters_from_checkpoint_file(
        checkpoint_file_path, filter_param_name)

    simple_replacements = [  # by priority
        ('densenet121/conv1', 'densenet_root_block/conv'),
        ('densenet121/logits', 'conv_logits'),
        ('densenet121', 'densenet_middle'),
        ('final_block/BatchNorm', 'post_bn_relu/bn'),
        ('weights', 'weights:0'),
        ('biases', 'bias:0'),
        ('beta', 'offset:0'),
        ('gamma', 'scale:0'),
    ]
    id = lambda x: x
    sub1 = lambda x: str(int(x) - 1)
    generic_replacements = [
        (r'dense_block(\d+)', ['db{}', sub1]),
        (r'conv_block(\d+)', ['block{}', sub1]),
        (r'x(\d+)\/BatchNorm', ['bn_relu{}/bn', sub1]),
        (r'x(\d+)\/Conv', ['conv{}', sub1]),
        (r'transition_block(\d+)\/blk\/BatchNorm',
         ['transition{}/bn_relu/bn', sub1]),
        (r'transition_block(\d+)\/blk\/Conv', ['transition{}/conv', sub1]),
    ]

    old_names = param_name_to_array.keys()
    new_names = translate_variable_names(old_names, simple_replacements,
                                         generic_replacements)
    old_name_to_new_name = dict(zip(old_names, new_names))
    return {
        nn: param_name_to_array[on]
        for on, nn in old_name_to_new_name.items()
    }


def get_ladder_densenet_parameters_from_checkpoint_file(checkpoint_file_path):
    # Checkpoint files can be downloaded here:
    # https://github.com/pudae/tensorflow-densenet
    param_name_to_array = get_densenet_parameters_from_checkpoint_file(
        checkpoint_file_path)

    param_name_to_array = { # without the last bn before logits 
        k: v
        for k, v in param_name_to_array.items() if "post_bn_relu" not in k
    }

    simple_replacements = [  # by priority
        ('densenet_root_block', 'ladder_densenet/densenet_root_block'),
        ('densenet_middle', 'ladder_densenet'),
    ]
    generic_replacements = [
        (r'db3/block([0-7])/', ['db3a/block{}/', lambda x: x]),
        (r'db3/block(\d+)/', ['db3b/block{}/', lambda x: str(int(x) - 8)]),
    ]

    new_names = translate_variable_names(
        param_name_to_array.keys(), simple_replacements, generic_replacements)

    name_to_new_name = dict(zip(param_name_to_array.keys(), new_names))
    return {
        name_to_new_name[n]: param_name_to_array[n]
        for n in param_name_to_array.keys()
    }
