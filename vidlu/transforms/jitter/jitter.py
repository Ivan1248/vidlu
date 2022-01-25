import dataclasses as dc
import inspect
import warnings
import typing as T
from functools import partial

import torch

from vidlu.utils.func import compose
import vidlu.transforms.image as vti
import vidlu.modules as vm
import vidlu.modules.pert as pert
import vidlu.data.types as dt

from .randaugment import RandAugment as PILRandAugment


class Composition:
    def __init__(self, *jitters):
        self.jitters = jitters

    def __call__(self, x):
        for f in self.jitters:
            x = f(x)
        return x


class PhTPS20:
    def __init__(self):
        from vidlu.training.robustness import perturbation as pert
        from ...modules.utils import init
        self.pert_model = pert.PhotoTPS20()
        self.init = init.MultiInit(
            dict(tps=init.NormalInit({'offsets': (0, 0.1)}),
                 photometric=init.UniformInit(
                     {'module.add_v.addend': [-0.25, 0.25],
                      'module.mul_s.factor': [0.25, 2.],
                      'module.add_h.addend': [-0.1, 0.1],
                      'module.mul_v.factor': [0.25, 2.]})))

    def __call__(self, inputs):
        (x, y), put_back = pick(lambda a: isinstance(a, dt.ArraySpatial2D), inputs)
        assert isinstance(x, dt.Image) and isinstance(y, dt.SegMap)
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
            x, y = self.pert_model((x, y))
        return put_back((x, y))


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
    raise NotImplementedError(f'Domain {type(x)} not supported by {tranform}.')


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


def pick(cond, inputs):
    # TODO: replace filt with indices?
    filt = {ind: i for i, ind in enumerate([i for i, x in enumerate(inputs) if cond(x)])}

    def put_back(outputs):
        return tuple(outputs[filt[i]] if i in filt else x for i, x in enumerate(inputs))

    return tuple(inputs[i] for i in filt), put_back


class SegRandScaleCropPadHFlip(pert.PertModel):
    domain = dt.Spatial2D
    supported = (dt.Image, dt.SegMap, dt.ClassAABBsOnImage)

    def __init__(self,
                 shape: tuple,
                 max_scale: float,
                 min_scale: float = None,
                 overflow: object = 0,
                 align_corners: bool = True,
                 image_pad_value: T.Union[torch.Tensor, float, T.Literal['mean']] = 'mean',
                 label_pad_value=-1,
                 random_scale_crop_f=vti.RandomScaleCrop,
                 scale_dist: vti.ScaleDistArg = "uniform"):
        super().__init__()
        # arguments should be in __dict__ because of modification by command line arguments
        self.__dict__.update(self.get_args(locals()))

    def forward(self, inputs):
        d, put_back = pick(lambda x: isinstance(x, dt.Spatial2D), inputs)
        d = self.random_scale_crop_f(
            shape=self.shape, max_scale=self.max_scale, min_scale=self.min_scale,
            overflow=self.overflow, align_corners=self.align_corners,
            scale_dist=self.scale_dist)(d)
        d = vti.RandomHFlip()(d)
        pad = partial(vti.pad_to_shape, shape=self.shape)
        d = [pad(x, value=self.image_pad_value) if isinstance(x, dt.Image) else
             pad(x, value=self.label_pad_value) if isinstance(x, dt.SegMap) else
             pad(x, value=0) if isinstance(x, dt.Spatial2D) else
             x
             for x in d]
        return put_back(d)

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
