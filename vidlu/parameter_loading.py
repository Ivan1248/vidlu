import torch

from vidlu.utils.collections import NameDict
from vidlu.utils import text


class ParameterNameTranslator:
    def __init__(self, input_output_format_pairs, input_must_match=True):
        self.input_output_format_pairs = tuple(input_output_format_pairs)
        self.translators = [text.FormatTranslator(inp, out, full_match=True)
                            for inp, out in input_output_format_pairs]
        self.input_must_match = input_output_format_pairs

    def __call__(self, input):
        for t in self.translators:
            output = t.try_translate(input)
            if output is not None:
                return output
        if not self.input_must_match:
            return input
        messages = []
        for (inp, out), t in zip(self.input_output_format_pairs, self.translators):
            try:
                t(input)
            except text.NoMatchError as ex:
                messages.append((inp, str(ex)))
        messages = '\n'.join(f"  {i}. {inp} -> {err}\n" for i, (inp, err) in enumerate(messages))
        raise RuntimeError(f'Input "{input}" matches no input format.\n{messages}')


def translate_dict_keys(dict_, input_output_format_pairs):
    translator = ParameterNameTranslator(input_output_format_pairs)
    return {translator(k): v for k, v in dict_.items()}


def remove_from_dict(dict_, patterns):
    raise NotImplementedError


def get_parameters(model_name, state_path):
    name_to_parameters = torch.load(state_path)
    return translate_parameters(model_name, name_to_parameters)


def translate_parameters(model_name, name_to_parameters):
    return globals()[f"translate_{model_name}_parameters"](name_to_parameters)


def translate_resnet_parameters(name_to_parameters):
    name_to_parameters = translate_dict_keys(
        name_to_parameters,
        {  # backbone
            "{a:conv|bn}1.{e}":
                "backbone.root.{a:bn->norm}.orig.{e}",
            "layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}.{e}":
                "backbone.features.unit{`int(a)-1`}_{b}.branching.block.{c:bn->norm}{`int(d)-1`}.orig.{e}",
            "layer{a:(\d+)}.{b:(\d+)}.downsample.{c:0|1}.{e:(.*)}":
                "backbone.features.unit{`int(a)-1`}_{b}.branching.shortcut.{c:0->conv|1->norm}.orig.{e}",
            # logits
            "fc.{e:(.*)}":
                "head.logits.orig.{e}"
        }.items())
    return name_to_parameters


def translate_densenet_parameters(name_to_parameters):
    name_to_parameters = translate_dict_keys(
        name_to_parameters,
        {  # backbone
            "features.{a:conv|norm}0.{e:(.*)}":
                "backbone.root.{a}.orig.{e}",
            "features.denseblock{a:(\d+)}.denselayer{b:(\d+)}.{c:conv|norm}{d:(\d+)}.{e}":
                "backbone.features.dense_block{`int(a)-1`}.unit{`int(b)-1`}.block.{c}{`int(d)-1`}.orig.{e}",
            "features.transition{a:(\d+)}.{b:conv|norm}.{e}":
                "backbone.features.transition{`int(a)-1`}.{b}.orig.{e}",
            "features.norm(\d+).{e:(.*)}":  # the number matching the (\d+) (1+number of blocks) is ignored
                "backbone.features.norm.orig.{e}",
            # logits
            "classifier.{e:(.*)}":
                "head.logits.orig.{e}"
        }.items())
    return name_to_parameters


def translate_swiftnet_parameters(name_to_parameters):
    name_to_parameters = translate_dict_keys(
        name_to_parameters,
        {  # backbone
            "backbone.{a:conv|bn}1{e}":
                "backbone.backbone.module.root.{a:bn->norm}.orig{e}",
            "backbone.layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}.{e}":
                "backbone.backbone.module.features.unit{`int(a)-1`}_{b}.branching.block.{c:bn->norm}{`int(d)-1`}.orig.{e}",
            "backbone.layer{a:(\d+)}.{b:(\d+)}.downsample.{c:0|1}.{e:(.*)}":
                "backbone.backbone.module.features.unit{`int(a)-1`}_{b}.branching.shortcut.{c:0->conv|1->norm}.orig.{e}",
            # spp
            "backbone.spp.spp.{a:spp_bn|spp_fuse}.{b:norm|conv}.{e:(.*)}":
                "backbone.context.{a:spp_bn->input_block|spp_fuse->fuse_block}.{b}0.orig.{e}",
            "backbone.spp.spp.spp{a:(\d+)}.{b:norm|conv}.{e:(.*)}":
                "backbone.context.pyramid.block{a}.{b}0.orig.{e}",
            # ladder
            "backbone.upsample.{a:(\d+)}.{b:bottleneck|blend_conv}.{c:norm|conv}.{e:(.*)}":
                "backbone.ladder.up_blends.{a}.{b:bottleneck->project|blend_conv->blend}.{c}0.orig.{e}",
            "logits.norm.{e:(.*)}":
                "backbone.norm.orig.{e}",
            # logits
            "logits.conv.{e:(.*)}":
                "head.logits.orig.{e}"
        }.items())
    class_count = name_to_parameters["head.logits.orig.weight"].shape[0]
    # requires_grad=False is ignored
    name_to_parameters["head.logits.orig.bias"] = torch.zeros(class_count)
    return name_to_parameters


def translate_swiftnet_orig_parameters(name_to_parameters):
    name_to_parameters = translate_dict_keys(
        name_to_parameters,
        {  # logits
            "logits.conv{e:(.*)}": "head.logits.orig{e}",
            # everything else
            "logits.norm{e:(.*)}": "backbone.norm{e}",
            "backbone.{e:(.*)}": "backbone.{e}"
        }.items())
    class_count = name_to_parameters["head.logits.orig.weight"].shape[0]
    # requires_grad=False is ignored
    name_to_parameters["head.logits.orig.bias"] = torch.zeros(class_count)
    return name_to_parameters
