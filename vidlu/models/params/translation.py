from warnings import warn
import torch

from vidlu.utils import text


def filter_by_and_remove_key_prefix(state_dict, prefix, error_on_no_match=False):
    if len(prefix) == 0:
        return state_dict
    result = {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if error_on_no_match and len(result) == 0:
        raise RuntimeError(
            f'No key in state_dict starts with {prefix=}. (Example keys: {list(state_dict)[:5]}.)')
    return result


def translate_dict_keys(dict_, input_output_format_pairs, context=None):
    translator = text.FormatTranslatorCascade(input_output_format_pairs, context=context)
    return {translator(k): v for k, v in dict_.items()}


def get_translated_parameters(translator_name, state_dict, subdict=''):
    subdict = [] if not subdict else subdict.split('/')
    for sub in subdict:
        state_dict = state_dict[sub]
    return translate(translator_name, state_dict)


def translate(translator_name, state_dict):
    return globals()[f"translate_{translator_name}"](state_dict)


# Translators ##################################################################


def translate_id(state_dict):
    return state_dict


def translate_madrylab_resnet(state_dict):
    start = 'module.model.'
    state_dict = {k[len(start):]: v for k, v in state_dict['model'].items() if k.startswith(start)}
    return translate_resnet(state_dict)


resnet_dict = {  # backbone
    r"{a:conv|bn}1{e}":
        r"backbone.root.{a:bn->norm}.orig{e}",
    r"layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}{e}":
        r"backbone.bulk.unit{`int(a)-1`}_{b}.fork.block.{c:bn->norm}{`int(d)-1`}.orig{e}",
    # logits
    r"fc{e}":
        r"head.logits.orig{e}",
}

resnet_v1_dict = {
    **resnet_dict,
    # backbone
    r"layer{a:(\d+)}.{b:(\d+)}.downsample.{c:0|1}{e}":
        r"backbone.bulk.unit{`int(a)-1`}_{b}.fork.shortcut.{c:0->conv|1->norm}.orig{e}",
}


def translate_resnet(state_dict):
    """Translates Torchvision ResNet parameters to Vidlu ResNet parameters."""
    return translate_dict_keys(state_dict, resnet_v1_dict.items())


resnet_v2_dict = {
    r"layer{a:([2-9]\d*)}.0.bn1{e}":  #
        r"backbone.bulk.unit{`int(a)-1`}_0.preact.norm0.orig{e}",
    **resnet_dict,
    # ResNet-v2 backbone
    r"layer{a:(\d+)}.{b:(\d+)}.downsample{e}":
        r"backbone.bulk.unit{`int(a)-1`}_{b}.fork.shortcut.orig{e}",
    r"bn5{e}":
        r"backbone.bulk.post_norm.orig{e}",
}


def translate_resnet_v2(state_dict):
    """Translates Torchvision ResNet parameters to Vidlu ResNet parameters."""
    return translate_dict_keys(state_dict, resnet_v2_dict.items())


def translate_densenet(state_dict):
    """Translates Torchvision Densenet parameters to Vidlu ResNet parameters."""
    return translate_dict_keys(
        state_dict,
        {  # backbone
            r"features.{a:conv|norm}0.{e:(.*)}":
                r"backbone.root.{a}.orig.{e}",
            r"features.denseblock{a:(\d+)}.denselayer{b:(\d+)}.{c:conv|norm}{d:(\d+)}.{e}":
                r"backbone.bulk.db{`int(a)-1`}.unit{`int(b)-1`}.fork"
                + r".block.{c}{`int(d)-1`}.orig.{e}",
            r"features.transition{a:(\d+)}.{b:conv|norm}.{e}":
                r"backbone.bulk.transition{`int(a)-1`}.{b}.orig.{e}",
            r"features.norm(\d+).{e:(.*)}":  # the number matching the (\d+) is ignored
                r"backbone.bulk.norm.orig.{e}",
            # logits
            r"classifier.{e:(.*)}":
                r"head.logits.orig.{e}"
        }.items())


def translate_swiftnet(state_dict):
    """Translates Marin's SwiftNet-RN parameters to Vidlu SwiftNet-RN
     parameters."""
    unused = ["backbone.img_std", "backbone.img_mean"]
    unused += [k for k in state_dict if k.startswith("criterion")]
    unused_arays = {k: state_dict[k] for k in unused}
    print(f"unused arrays: {unused_arays}")
    for k in unused:
        del state_dict[k]
    state_dict = translate_dict_keys(
        state_dict,
        {  # backbone
            r"backbone.{a:conv|bn}1{e}":
                r"backbone.backbone.root.{a:bn->norm}.orig{e}",
            r"backbone.layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}.{e}":
                r"backbone.backbone.bulk.unit{`int(a)-1`}_{b}.fork"
                + r".block.{c:bn->norm}{`int(d)-1`}.orig.{e}",
            r"backbone.layer{a:(\d+)}.{b:(\d+)}.downsample.{c:0|1}.{e:(.*)}":
                r"backbone.backbone.bulk.unit{`int(a)-1`}_{b}.fork"
                + r".shortcut.{c:0->conv|1->norm}.orig.{e}",
            # spp
            r"backbone.spp.spp.{a:spp_bn|spp_fuse}.{b:norm|conv}.{e:(.*)}":
                r"backbone.context.{a:spp_bn->input_block|spp_fuse->fuse_block}.{b}0.orig.{e}",
            r"backbone.spp.spp.spp{a:(\d+)}.{b:norm|conv}.{e:(.*)}":
                r"backbone.context.pyramid.block{a}.{b}0.orig.{e}",
            # ladder
            r"backbone.upsample.{a:(\d+)}.{b:bottleneck|blend_conv}.{c:norm|conv}.{e:(.*)}":
                r"backbone.ladder.up_blends.{a}.{b:bottleneck->project|blend_conv->blend}"
                + r".{c}0.orig.{e}",
            r"logits.norm.{e:(.*)}":
                r"backbone.norm.orig.{e}",
            # logits
            r"logits.conv.{e:(.*)}":
                r"head.logits.orig.{e}"
        }.items())
    return state_dict


def translate_swiftnet_orig_backbone(state_dict):
    state_dict = translate_dict_keys(
        state_dict,
        {  # logits
            r"logits.conv{e:(.*)}": "head.logits.orig{e}",
            # everything else
            r"logits.norm{e:(.*)}": "backbone.norm{e}",
            r"backbone.{e:(.*)}": "backbone.{e}"
        }.items())
    return state_dict
