import dataclasses as dc
import inspect
import warnings
import typing as T
from functools import partial

import torch
import torchvision.transforms as tvt

from vidlu.utils.func import compose
import vidlu.transforms.image as vti
import vidlu.modules as vm
import vidlu.modules.pert as pert
import vidlu.modules.utils.init as vmui
import vidlu.data.types as dt

from .randaugment import RandAugment as PILRandAugment


class Composition:
    def __init__(self, *jitters):
        self.jitters = jitters

    def __call__(self, x):
        for f in self.jitters:
            x = f(x)
        return x


class SegmentationJitter(pert.PertModel):
    pass


class ClassificationJitter:
    def __call__(self, x):
        if len(x) == 1:
            return (self.apply_input(x[0]),)
        return self.apply_input(x[0]), self.apply_label(x[1])

    def apply_input(self, x):
        return x

    def apply_label(self, y):
        return y


@dc.dataclass()
class ReplaceWithNoise(ClassificationJitter):
    mean: float = 0.
    std: float = 1.

    def apply_input(self, x):
        return torch.randn_like(x) * self.std + self.mean


####################################################################################################

def _args_repr(obj):
    args = {k: getattr(obj, k) for k in inspect.signature(type(obj)).parameters.keys()}
    args_str = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
    return f'{type(obj).__name__}({args_str})'


def unsupported_error(tranform, x):
    raise NotImplementedError(f'Modality {type(x)} not supported by {tranform}.')


class CifarPadRandCropHFlip(ClassificationJitter):
    def apply_input(self, x):
        return compose(vti.Pad(4), vti.RandomCrop(x[0].shape[-2:]), vti.RandomHFlip())(x)


class SegRandHFlip(SegmentationJitter):
    def apply(self, x):
        return vti.RandomHFlip()(tuple(x))


@dc.dataclass
class SegRandCropHFlip(SegmentationJitter):
    crop_shape: tuple

    def apply(self, x):
        return compose(vti.RandomCrop(self.crop_shape), vti.RandomHFlip())(tuple(x))


def pick_out(cond: T.Callable, inputs: tuple) -> T.Tuple[tuple, T.Callable[[tuple], tuple]]:
    # Dictionary to track indices of filtered elements
    filtered_indices = {ind: i for i, ind in enumerate(i for i, x in enumerate(inputs) if cond(x))}

    def put_back_together(outputs: tuple) -> tuple:
        # Reinsert filtered elements into their original positions
        return tuple(outputs[filtered_indices[i]] if i in filtered_indices else x
                     for i, x in enumerate(inputs))

    # Tuple of filtered elements and the function for putting the tuple back together
    return tuple(inputs[i] for i in filtered_indices), put_back_together


def only_applies_to(type):
    def decorator(func):
        def wrapper(inputs):
            d, put_back = pick_out(lambda x: isinstance(x, type), inputs)
            d = func(d)
            return put_back(d)

        return wrapper

    return decorator


class SegRandScaleCropPadHFlip(pert.PertModel):
    domain = dt.Spatial2D
    supported = (dt.Image, dt.SegMap, dt.ClassAABBsOnImage)

    def __init__(self,
                 shape: tuple,
                 max_scale: float,
                 min_scale: float = None,
                 scale_factor_fn=lambda h, w: 1.,
                 overflow: object = 0,
                 align_corners: bool = True,
                 image_pad_value: T.Union[torch.Tensor, float, T.Literal['mean']] = 'mean',  # 0.?
                 label_pad_value=-1,
                 random_scale_crop_f=vti.RandomScaleCrop,
                 scale_dist: vti.ScaleDistArg = "uniform"):
        super().__init__()
        # arguments should be in __dict__ because of modification by command line arguments
        self.__dict__.update(self.get_args(locals()))

    def forward(self, inputs):
        @only_applies_to(dt.Spatial2D)
        def inner_forward(d: T.Sequence[dt.Spatial2D]):
            d = self.random_scale_crop_f(
                shape=self.shape, max_scale=self.max_scale, min_scale=self.min_scale,
                overflow=self.overflow, scale_factor_fn=self.scale_factor_fn,
                align_corners=self.align_corners, scale_dist=self.scale_dist)(d)
            d = vti.RandomHFlip()(d)
            pad = partial(vti.pad_to_shape, shape=self.shape)
            d = [pad(x, value=self.image_pad_value) if isinstance(x, dt.Image) else
                 pad(x, value=self.label_pad_value) if isinstance(x, dt.SegMap) else
                 pad(x, value=0) if isinstance(x, dt.Spatial2D) else
                 x
                 for x in d]
            return d

        return inner_forward(inputs)

    def __repr__(self):
        return _args_repr(self)


class RandAugment(ClassificationJitter):
    def __init__(self, n, m):
        self.pil_randaugment = PILRandAugment(n, m)

    def apply_input(self, x):
        with torch.no_grad():
            warnings.warn("RandAugment is not differentiable.")
            h = x.mul(255).to(torch.uint8)
            h = vti.torch_to_pil(h.permute(1, 2, 0).cpu())
            h = self.pil_randaugment(h)
            return vti.pil_to_torch(h).to(dtype=x.dtype, device=x.device).permute(2, 0, 1) / 256


class PertModelWrapper:
    def __init__(self, pert_model, init):
        self.pert_model = pert_model
        self.init = init

    def __call__(self, inputs):
        (x, y), put_back = pick_out(lambda a: isinstance(a, (dt.ArraySpatial2D, dt.ClassLabel)),
                                    inputs)

        x_ = x.unsqueeze(0)

        with torch.no_grad():
            if not vm.is_built(self.pert_model):
                self.pert_model(x_)
                for p in self.pert_model.parameters():
                    p.requires_grad = False
            self.init(self.pert_model, x_)

        if isinstance(y, dt.ClassLabel):
            x = self.pert_model(x_).squeeze(0)
        else:
            x_, y_ = self.pert_model(x_, y.unsqueeze(0))
            x = x_.squeeze(0)
            y = y_.squeeze(0)

        return put_back((x, y))


from vidlu.training.robustness import perturbation as pert
import vidlu.modules.utils as vmu


class PhTPS20(PertModelWrapper):
    def __init__(self):
        pert_model = pert.PhotoTPS20()
        init = vmu.MultiInit(
            dict(tps=vmui.NormalInit({'offsets': (0, 0.1)}),
                 photometric=vmui.UniformInit(
                     {'add_v': [-0.25, 0.25],
                      'mul_s': [0.25, 2.],
                      'add_h': [-0.1, 0.1],
                      'mul_v': [0.25, 2.]})))
        super().__init__(pert_model, init)


class TPS20(PertModelWrapper):
    def __init__(self):
        super().__init__(vmi.BackwardTPSWarp(label_interpolation_mode='nearest',
                                             label_padding_mode=-1),
                         vmu.NormalInit({'offsets': (0, 0.1)}))


class Ph20(PertModelWrapper):
    def __init__(self):
        pert_model = pert.Photometric20(forward_arg_count=1)
        init = vmu.UniformInit(
            {'add_v': [-0.25, 0.25],
             'mul_s': [0.25, 2.],
             'add_h': [-0.1, 0.1],
             'mul_v': [0.25, 2.]})
        super().__init__(pert_model, init)


class Photometric3(PertModelWrapper):
    def __init__(self):
        pert_model = pert.Photometric3(clamp=True)
        init = vmu.UniformInit(
            {'add_h.addend': [-1 / 6, 1 / 6],
             'add_s.addend': [-0.2, 0.2],
             'add_v.addend': [-0.1, 0.1],
             'mul_v.factor': [0.25, 2.]})
        super().__init__(pert_model, init)


class ColorJitter:
    def __init__(self, brightness: T.Union[float, T.Tuple[float, float]] = 0,
                 contrast: T.Union[float, T.Tuple[float, float]] = 0,
                 saturation: T.Union[float, T.Tuple[float, float]] = 0,
                 hue: T.Union[float, T.Tuple[float, float]] = 0):
        self.pert = tvt.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                    hue=hue)

    def __call__(self, inputs):
        return [self.pert(x) if isinstance(x, dt.Image) else x for x in inputs]
