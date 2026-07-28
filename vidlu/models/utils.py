import warnings

import torch.nn.functional as F

import vidlu.modules.components as vmc
import vidlu.modules as vm


def pad_to_multiple(x, divisor, value=0.0):
    """Pads the last two (spatial) dimensions of `x` on the bottom/right up to a multiple
    of `divisor` (the mmsegmentation `size_divisor` convention).

    Returns `(x_padded, (h, w))`, where `(h, w)` is the original spatial size, so a model
    output computed on `x_padded` can be cropped back with `output[..., :h, :w]`. When `x`
    is already divisible, it is returned unchanged.
    """
    h, w = x.shape[-2:]
    ph, pw = -(-h // divisor) * divisor, -(-w // divisor) * divisor
    if (ph, pw) == (h, w):
        return x, (h, w)
    return F.pad(x, (0, pw - w, 0, ph - h), value=value), (h, w)


def _last_by_prefix(names, name_to_prefix, parent_last):
    prev_name = names[0]
    for name in names[1:]:
        if (a := name_to_prefix(name)) != (b := name_to_prefix(prev_name)):
            yield prev_name
        if not (parent_last and name.startswith(prev_name + '.')):
            prev_name = name
    yield prev_name


def ladder_input_names(backbone):
    if isinstance(backbone, (vmc.ResNetV1Backbone, vmc.ResNetV2Backbone, vmc.IRevNetBackbone)):
        prefix_end = '_'  # bulk.unit_{i}_{j}
        names = [name for name in next(zip(*backbone.bulk.named_modules())) if name.startswith("unit")]
        names = _last_by_prefix(names, name_to_prefix=lambda name: name.split(prefix_end, 1)[0],
                                parent_last=True)
    elif isinstance(backbone, vmc.DenseNetBackbone):
        names = [name for name in next(zip(*backbone.bulk.named_children())) if name.startswith("db")]
    return [f"bulk.{name}" for name in names][:-1]


def set_all_inplace(module, inplace):
    for name, module in module.named_modules():
        if isinstance(module, vm.Sum) or hasattr(module, 'inplace'):
            if module.inplace and not inplace:
                warnings.warn(f"`inplace` attribute of module {name} overridden with `False`.")
            module.inplace = inplace  # ResNet-10: 8312MiB, 6.30/s -> 6734MiB, 6.32/s
