import functools
import typing as T
from vidlu.utils.func import partial
import copy
from warnings import warn

import torch
from torch import nn
import torch.nn.functional as F
from numpy import s_
from typeguard import check_argument_types

from vidlu.torch_utils import round_float_to_int
import vidlu.modules.elements as E
import vidlu.modules.functional as vmf
import vidlu.modules.components as vmc
import vidlu.ops.image as voi
import vidlu.utils.func as vuf
import vidlu.data.types as dt
from vidlu.data import Record
import vidlu.utils.text as vut


class BatchParameter(torch.nn.Parameter):
    r"""A kind of Tensor that is to be considered a batch of parameters, i.e.
    each input example has its own parameter.

    It is used for make perturbation modules slicable in the batch dimension.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def __repr__(self):
        return f'{type(self).__name__}:\n{repr(self.data)}'

    def __reduce_ex__(self, proto):
        param = BatchParameter(self.data, self.requires_grad)
        param._backward_hooks = dict()
        return param


def _get_param(x, factory_or_value):
    factory = (factory_or_value if callable(factory_or_value)
                                   and not isinstance(factory_or_value, torch.Tensor)
               else partial(torch.full, fill_value=factory_or_value))
    return factory(x.shape, device=x.device)


def _complete_shape(shape_tail, input_shape):
    return input_shape[:-len(shape_tail)] + tuple(
        b if a is None else a for a, b in zip(shape_tail, input_shape[-len(shape_tail):]))


_lower_to_upper_type_name = {k.lower() for k in dir(dt) if k[0].isupper()}


def _name_to_type(name):
    class_name = vut.to_pascal_case(name)
    try:
        return getattr(dt, class_name)
    except AttributeError as e:
        msg = f'The key "{name}" corresponds to non-existing class name "{class_name}". '
        if (class_name_ := _lower_to_upper_type_name.get(name.lower(), None)) is None:
            raise type(e)(msg + f"{e.args[0]}", *e.args[1:])
        else:
            msg += f"Parhaps the key should be {vut.to_snake_case(class_name_)}. "
            raise type(e)(msg + f"{e.args[0]}", *e.args[1:])


def to_typed_args(**kwargs):
    return tuple(getattr(dt, _name_to_type(k))(v) for k, v in kwargs)


class PertModel(E.Module):
    domain: T.Optional[T.Union[T.Tuple[type], type]] = dt.Domain
    supported: T.Optional[T.Union[T.Tuple[type], type]] = dt.Domain
    param_defaults = dict()

    def __init__(self, domain=None):
        super().__init__()
        if domain is not None:
            self.domain = domain

    def _check_default_params(self, default_params):
        if (dp := set(default_params)) != (pd := set(self.param_defaults)):
            raise TypeError(f"Names of parameters returned by `create_default_params` ({dp}) do"
                            f" not match expected names from `param_defaults` ({pd}).")

    def build(self, inputs):
        # dummy_x has properties like x, but takes up almost no memory
        x = next(a for a in inputs if isinstance(a, torch.Tensor))
        self.dummy_x = type(x)(x.new_zeros(()).expand(x.shape))
        default_params = self.create_default_params(x)
        self._check_default_params(default_params)
        for k, v in default_params.items():
            setattr(self, k, BatchParameter(v, requires_grad=True))

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, (dict, Record)):
            inputs_tuple = [(getattr(dt, vut.to_pascal_case(dom), lambda x: x))(v)
                            for dom, v in inputs.items()]
        else:
            inputs_tuple = inputs

        if any(type(x) is torch.Tensor for x in inputs_tuple):
            raise TypeError(
                f"torch.Tensor inputs should be instances of {dt.Domain.__qualname__}.")
        for x in inputs_tuple:
            if isinstance(x, self.domain) and not isinstance(x, self.supported):
                raise NotImplementedError(
                    f'Inputs of type {type(x)} are not supported by ({type(self)}).')
        result_tuple = super().__call__(inputs_tuple, **kwargs)

        if not (tin := tuple(map(type, inputs_tuple))) == (tout := tuple(map(type, result_tuple))):
            raise TypeError(f'Output types of do not match input types ({tin} != {tout}).')

        return result_tuple if inputs_tuple is inputs else type(inputs)(
            zip(inputs.keys(), result_tuple))

    def forward(self, inputs, **kwargs):
        def error(x):
            raise TypeError(f"Input domain {type(x)} not supported.")

        return tuple(self.forward_single(x, **kwargs) if isinstance(x, self.domain) else error(x)
                     for x in inputs)

    def forward_single(self, input, **kwargs):
        raise TypeError(
            f"Either forward(*input, **kwargs) or forward_single(inputs, **kwargs) has to be"
            + f" implemented for type {type(self)}.")

    def create_default_params(self, x):
        return dict()

    def default_parameters(self, full_size: bool, recurse=True):
        r"""Returns an iterator over default module parameters.

        Args:
            full_size: If False, arrays (or scalars) of the minimum shape
                necessary to compute the difference of parameters to their
                default values are yielded. This can be useful for constraints
                or regularization. If True, parameters like those used for
                initialization of the module are yielded.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only default parameters
                for this module.

        Yields:
            Parameter: module parameter
        """
        for _, param in self.named_default_parameters(full_size=full_size, recurse=recurse):
            yield param

    def _named_members_repeatable(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules.
        Modified so that memo contains names instead of values, which may not be
        unique."""
        memo = set()  # Set containing names instead of values
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                name = module_prefix + ('.' if module_prefix else '') + k
                if name in memo:
                    continue
                memo.add(name)
                yield name, v

    def named_default_parameters(self, full_size: bool, prefix='', recurse=True):
        r"""Returns an iterator over just created default module parameters,
        yielding both the name of the default parameter and the parameter.

        Args:
            full_size: If False, arrays (or scalars) of the minimum shape
                necessary to compute the difference of parameters to their
                default values are yielded. This can be useful for constraints
                or regularization. If True, parameters like those used for
                initialization of the module are yielded.
            prefix (str): Prefix to prepend to all parameter names.
            recurse (bool): If True, then yields default parameters of this
                module and all submodules. Otherwise, yields only parameters of
                this module.

        Yields:
            (string, Tensor): Tuple containing the name and parameter
        """
        getpd = ((lambda m: m.create_default_params(m.dummy_x).items()) if full_size
                 else (lambda m: ((k, v['value']) for k, v in m.param_defaults.items())))
        return self._named_members_repeatable(
            lambda m: getpd(m) if isinstance(m, PertModel) else iter(()),
            prefix=prefix, recurse=recurse)

    def reset_parameters(self):
        for (name, data), (name_, def_value) in zip(self.named_parameters(),
                                                    self.named_default_parameters(True)):
            assert name == name_
            data.set_(def_value if not callable(def_value)
                                   and not isinstance(def_value, torch.Tensor) else
                      def_value())

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        warn(f"ensure_output_within_bounds is called"
             + f" but not implemented for {type(self).__name__}")


def default_parameters(pert_model, full_size: bool, recurse=True):
    return PertModel.default_parameters(pert_model, full_size, recurse=recurse)


def named_default_parameters(pert_model, full_size: bool, recurse=True):
    return PertModel.named_default_parameters(pert_model, full_size, recurse=recurse)


def reset_parameters(pert_model):
    PertModel.reset_parameters(pert_model)


def _blend(x1, x2, ratio):
    return (ratio * x1).add_((1.0 - ratio) * x2)


def _pert_model_forward(module, *args, **kwargs):
    assignable, other = vuf.pick_assignable_args(
        module.forward if isinstance(module, nn.Module) else module, kwargs, return_other=True,
        kwargs_policy="pick")
    return dict(**module(*args, **assignable), **other)


"""
class PertModelWrapper(PertModelBase):
    def __init__(self, module, arg_types: T.Sequence = None):
        # T.Union[T.Sequence, T.Mapping, ArgHolder]
        super().__init__()
        self.module = module
        self.arg_types = arg_types
        # self.arg_types = (arg_types if isinstance(arg_types, (ArgHolder, type(None))) else
        #                   ArgHolder(*arg_types) if isinstance(arg_types, T.Sequence) else
        #                   ArgHolder(**arg_types))

    def forward(self, *args):
        # input_arg_types = ArgHolder(*tuple(type(x) for x in args),
        #                             **{k: type(v) for k, v in kwargs.items()})
        index_mapping = {i: self.arg_types.index(type(x)) for i, x in enumerate(args)}
        args_inner = [None] * len(self.arg_types)
        for i_in, i_inner in index_mapping.items():
            args_inner[i_inner] = args[i_in]
        result_inner = self.module(*args_inner)
        result = [None] * len(args)
        for i_in, i_inner in index_mapping.items():
            result[i_in] = result_inner[i_inner]
        return result
"""


def single_dispatch_func(*funcs):
    func = funcs[0]

    @functools.singledispatch
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    for f in funcs[1:]:
        wrapper.register(f)
    return wrapper


class FuncPertModel(PertModel):
    def __init__(self, func, single: bool):
        super().__init__()
        self.func = func
        self.single = single

    def forward(self, inputs, **kwargs):
        return self.func(inputs, **kwargs) if not self.single else super().forward(inputs, **kwargs)

    def forward_single(self, input, **kwargs):
        return self.func(input, **kwargs)


class EquivariantPertModel(PertModel):
    param_defaults = dict()

    def __init__(self, equivariant_dims: T.Sequence, **kwargs):
        super().__init__(**kwargs)
        self.equivariant_dims = equivariant_dims

    def create_default_params(self, x):
        shape = list(x.shape)
        for d in self.equivariant_dims:
            shape[d if d >= 0 else len(x.shape) - d] = 1
        dummy = x.new_zeros(()).expand(shape)  # contains shape, dtype, and device
        return {k: _get_param(dummy, self.param_defaults[k]['value'])
                for k, v in self.param_defaults.items()}


class Space2dEquivariantPertModel(PertModel):
    param_defaults = dict()

    def __init__(self, equivariant_dims: T.Sequence, **kwargs):
        super().__init__(**kwargs)
        self.equivariant_dims = equivariant_dims

    def create_default_params(self, x):
        shape = list(x.shape)
        for d in self.equivariant_dims:
            shape[d if d >= 0 else len(x.shape) - d] = 1
        dummy = x.new_zeros(()).expand(shape)  # contains shape, dtype, and device
        return {k: _get_param(dummy, self.param_defaults[k]['value'])
                for k, v in self.param_defaults.items()}


class SliceEquivariantPertModel(PertModel):
    """Can modify only a slice, e.g. a channel."""
    param_defaults = dict()

    def __init__(self, equivariant_dims: T.Sequence, slice=None,
                 **kwargs):  # slice=np.s_[...] can be used
        super().__init__(**kwargs)
        self.equivariant_dims = equivariant_dims
        self.slice = slice if slice is None or isinstance(slice, tuple) else (slice,)

    def create_default_params(self, x):
        if self.slice is not None:
            x = x[self.slice]
        shape = list(x.shape)
        for d in self.equivariant_dims:
            shape[d if d >= 0 else len(x.shape) - d] = 1
        dummy = x.new_zeros(()).expand(shape)  # contains shape, dtype, and device
        return {k: _get_param(dummy, self.param_defaults[k]['value'])
                for k, v in self.param_defaults.items()}


def apply_batch_slice(pmodel, slice_):
    orig_params = dict(pmodel._parameters)
    pmodel.orig_params = orig_params
    for k, p in orig_params.items():
        pmodel[k] = p[slice_]


def undo_batch_slice(pmodel):
    pmodel._parameters.update(pmodel._orig_params)
    del pmodel.orig_params


def call_on_batch_slice(pmodel, x, slice_):
    apply_batch_slice(pmodel, slice_)
    y = pmodel(x[slice_])
    undo_batch_slice(pmodel)
    return y


class AlterGamma(Space2dEquivariantPertModel):
    # domain = (vdt.Image,)
    param_defaults = dict(gamma=dict(value=1., bounds=[0, 500]))
    eps = 1e-8

    def forward_single(self, x):
        if isinstance(x, dt.Image):
            return x.mul_(1 - 2 * self.eps).add_(self.eps).pow(self.gamma)
        elif isinstance(x, (dt.SegMap, dt.ClassLabelLike)):
            return x
        else:
            raise TypeError(f"Unsupported input type {type(x).__qualname__}.")


class AlterLogGamma(Space2dEquivariantPertModel):
    # Gradients are more stable than for AlterGamma
    param_defaults = dict(log_gamma=dict(value=0., bounds=[-6, 6]))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        return x.pow(self.log_gamma.exp())


def _get_center(center_arg, x, equivariant_dims):
    return (x.mean(equivariant_dims, keepdim=True) if center_arg == 'mean' else
            center_arg(x) if callable(center_arg) else center_arg)


class Contrast(Space2dEquivariantPertModel):
    """Linearly interpolates or extrapolates between inputs and centers.

    The interpolation factor (contrast) can be any positive number.

    To make it work like adjust_contrast in Torchvision, center should be set to
    the average luma value across the whole image according to CCIR 601:

    >>> from vidlu.ops.image import rgb_to_luma
    >>> center = lambda x: rgb_to_luma(x, 601).unsquezze(-3).mean((-2, -1), keepdim=True)
    >>> torchvision_contrast = Contrast(center=center)
    """
    param_defaults = dict(factor=dict(value=1., bounds=[0, float('inf')]))

    def __init__(self, equivariant_dims=(-2, -1),
                 center: T.Union[float, T.Callable, T.Literal['mean']] = 0.5):
        check_argument_types()
        super().__init__(equivariant_dims)
        self.center = center

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        center = _get_center(self.center, x, equivariant_dims=self.equivariant_dims)
        return _blend(x, center, self.factor)


class TorchvisionSaturation(Contrast):
    param_defaults = dict(factor=dict(value=1., bounds=[0, float('inf')]))

    def __init__(self, greyscale_f=partial(voi.rgb_to_luma, rev=601)):
        super().__init__(equivariant_dims=())
        self.greyscale_f = greyscale_f

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        grey = self.greyscale_f(x)
        if len(grey.shape) < len(x.shape):
            grey = grey.unsqueeze(-3)
        return _blend(x, grey, self.factor)


class Add(SliceEquivariantPertModel):
    param_defaults = dict(addend=dict(value=0., bounds=[-1, 1]))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        if self.slice is None:
            return x + self.addend
        y = x.clone()
        y[self.slice] += self.addend
        return y

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        if len(self.equivariant_dims) == 0:
            self.addend.add_(x).clamp_(*bounds).sub_(x)
        else:
            raise NotImplementedError


class Multiply(SliceEquivariantPertModel):
    param_defaults = dict(factor=dict(value=1., bounds=[0, float('inf')]))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        if self.slice is None:
            return x * self.factor
        y = x.clone()
        y[self.slice] *= self.factor
        return y

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        if len(self.equivariant_dims) == 0:
            self.factor.mul_(x).clamp_(*bounds).div_(x)
        else:
            raise NotImplementedError


class Modulo(SliceEquivariantPertModel):
    param_defaults = dict(period=dict(value=1., bounds=[0, float('inf')]))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        if self.slice is None:
            return x % self.factor
        y = x.clone()
        y[self.slice] %= self.period
        return y


class Whiten(Space2dEquivariantPertModel):
    """Interpolates pixel values between the original ones and 1."""
    param_defaults = dict(weight=dict(value=0., bounds=[0, 1]))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        return ((1 - self.weight) * x).add_(self.weight)


class AlterColor(PertModel):
    """Applies a transformation similar to ColorJitter in Torchvision.
    """
    param_defaults = dict(
        order=dict(value=lambda **k: torch.arange(len(k['x']), len(k['self'].module)),
                   bounds=[0, 1]))

    def __init__(self, brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                 hue=(-0.5, 0.5)):
        super().__init__()
        modules = dict(brightness=Multiply(equivariant_dims=(-1, -2)),
                       contrast=Contrast(center='mean'),
                       saturation=TorchvisionSaturation(),
                       hue=E.Seq(to_hsv=FuncPertModel({dt.Image: voi.rgb_to_hsv}),
                                 add_h=Add((2, 3), slice=s_[:, 0:, ...]),  # no need for modulo
                                 to_rgb=FuncPertModel({dt.Image: voi.hsv_to_rgb})))
        self.modules = E.ModuleTable({k: v for k, v in modules.items() if locals()[k] is not None})
        self.order = None

    def build(self, x):
        super().build(x)
        for m in self.module:
            m(x)

    def initialize(self, brightness=lambda x: x.uniform_(0.5, 1.5),
                   contrast=lambda x: x.uniform_(0.5, 1.5),
                   saturation=lambda x: x.uniform_(0.5, 1.5), hue=lambda x: x.uniform_(-0.5, 0.5)):
        for k, m in self.module.named_children():
            if k != 'hue':
                locals()[k](m.factor)
        if 'hue' in self.module:
            hue(self.module.hue.add_h.addend)
        self.order = torch.stack(torch.randperm(len(self.module)))

    def forward_single(self, x):
        if not isinstance(x, dt.Image):
            return x
        n = len(self.module)
        modules = list(self.module)
        for i in range(n):
            r = x.new_empty()
            for j, m in enumerate(modules):
                slice_ = self.order[:, i] == j
                r[slice_] = call_on_batch_slice(m, x, slice_)
            x = r
        return x


# Warps ############################################################################################


def angle_preserving_matrix_2d(ang, scale):
    matrix = torch.eye(2, 2)
    sin_ang = ang.sin()
    cos_ang = ang.cos()
    matrix[:, 0, 0] = matrix[:, 1, 1] = cos_ang
    matrix[:, 0, 1] = -sin_ang
    matrix[:, 1, 0] = sin_ang
    matrix *= scale
    return matrix


class AnglePreservingLinear2d(PertModel):
    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True):
        breakpoint()  # TODO
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def create_default_params(self, x):
        return dict(scale=x.new_ones(x.shape[0]),
                    ang=x.new_zeros(x.shape))

    def forward_single(self, x):
        if not isinstance(x, dt.Spatial2D):
            return x
        matrix = angle_preserving_matrix_2d(self.ang, self.scale)

    def inverse_module(self):
        inv = copy.deepcopy(self)
        with torch.no_grad:
            inv.scale.set_(1 / inv.scaling)
            inv.ang.neg_()


_no_default = object()


def _subtype_key_get(dict_, type_key, default=_no_default):
    for supertype in type_key.__mro__:
        if supertype in dict_:
            return dict_[supertype]
    return default if default is not _no_default else dict_[type_key]


class TranslationSpatial(PertModel):
    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True,
                 params: T.Mapping[type, T.Mapping] = None):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        self.params = params or dict()

    def create_default_params(self, x):
        return dict(translation=x.new_zeros(x.shape))

    def forward_single(self, x):
        if not isinstance(x, dt.Spatial2D):
            return x
        return vmf.warp(x, self.flow,
                        **{**self.args, **_subtype_key_get(self.params, type(x), dict())})

    def inverse_module(self):
        inv = copy.deepcopy(self)
        with torch.no_grad:
            inv.translation.set_(-inv.translation)


class Warp(PertModel):
    param_defaults = dict(flow=dict(value=0., bounds=[-1, 1]))

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def create_default_params(self, x):
        return dict(flow=x.new_zeros((x.shape[0], 2, *x.shape[2:])))

    def forward_single(self, x):
        if not isinstance(x, dt.Spatial2D):
            return x
        return vmf.warp(x, self.flow, **self.args)


class SmoothWarp(PertModel):
    param_defaults = dict(unsmoothed_flow=dict(value=0., bounds=[-100, 100]))

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True,
                 smooth_f=partial(vmc.GaussianFilter2D, sigma=3)):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        self.smooth = smooth_f()

    def create_default_params(self, x):
        return dict(unsmoothed_flow=x.new_zeros((x.shape[0], 2, *x.shape[2:])))

    def forward_single(self, x):
        if not isinstance(x, dt.Spatial2D):
            return x
        flow = self.smooth(self.unsmoothed_flow)
        return vmf.warp(x, flow, **self.args)


def grid_sample_p(x, grid, mode, padding_mode, align_corners):
    pm = 'zeros' if padding_mode == 0 else padding_mode
    if isinstance(pm, str):
        return F.grid_sample(x, grid, mode=mode, padding_mode=pm,
                             align_corners=align_corners).squeeze_(1)
    else:
        return F.grid_sample(x - pm, grid, mode=mode, padding_mode='zeros',
                             align_corners=align_corners).squeeze_(1).add_(pm)


def _forward_softsplat_warp(x, grid, mode, padding_mode, align_corners):
    # https://github.com/sniklaus/softmax-splatting
    assert align_corners == True
    from vidlu.ops import softmax_splatting
    pv = 0 if padding_mode in (0, 0., "zeros") else padding_mode
    warn("align_corners and mode not used in _forward_warp")
    H, W = grid.shape[-3:-1]
    base_grid = vmf.uniform_grid_2d((H, W), low=-1., high=1., device=grid.device, dtype=grid.dtype)
    offsets = grid - base_grid
    flow = offsets.permute(0, 3, 1, 2).mul_(offsets.new([W / 2, H / 2]).view(1, 2, 1, 1))
    result = softmax_splatting.FunctionSoftsplat(x if pv == 0 else x - pv, flow.contiguous(),
                                                 tenMetric=None, strType="average")
    return result if pv == 0 else result.add_(pv)


def warp(x, grid, mode='bilinear', padding_mode='zeros',
         align_corners=True, forward=False):
    _warp_func = _forward_softsplat_warp if forward else grid_sample_p

    continuous = 'float' in f'{x.dtype}'
    x_p = (x if continuous else x.to(grid.dtype))
    single_channel = x.dim() == 3
    x_p = _warp_func(x_p[:, None, ...] if single_channel else x_p, grid, mode=mode,
                     padding_mode=padding_mode, align_corners=align_corners)
    if single_channel:
        x_p.squeeze_(1)
    if x_p.dtype is not x.dtype:
        x_p = round_float_to_int(x_p, x.dtype)
    return x_p


def _warp_tuple(inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
                label_mode='nearest', label_padding_mode=-1, forward=False):
    warp_ = partial(warp, grid=grid, align_corners=align_corners, forward=forward)
    res = []
    for x in inputs:
        if x.dim() < 3:
            res.append(x)
        elif isinstance(x, dt.SegMap):
            res.append(warp_(x, mode=label_mode, padding_mode=label_padding_mode))
        elif 'float' in str(x.dtype):
            res.append(warp_(x, mode=mode, padding_mode=padding_mode))
        else:
            raise TypeError(
                f"Unsupported input with type={type(x)}, shape={x.shape}, dtype={x.dtype}.")
    return tuple(res)


class TPSWarp(PertModel):
    param_defaults = dict(offsets=dict(value=0., bounds=[-0.2, 0.2]))

    def __init__(self, *, forward: bool, control_grid_shape=(2, 2),
                 control_grid_align_corners=False, align_corners=True, padding_mode='zeros',
                 interpolation_mode='bilinear', label_interpolation_mode=None,
                 label_padding_mode=None, swap_src_dst=False, center_offsets=False):
        # control_grid_align_corners=False puts control points into centers of parts of the subdivided image
        super().__init__()
        self.args = self.get_args(locals())
        self.warp_args = {
            k: self.args.pop(k)
            for k in ['interpolation_mode', 'padding_mode', 'label_interpolation_mode',
                      'label_padding_mode', 'forward']}

    def build(self, x):
        k = dict(device=x.device, dtype=x.dtype)
        with torch.no_grad():
            cgs = self.args.control_grid_shape
            self.c_src = vmf.uniform_grid_2d(cgs, **k).view(-1, 2)
            if not self.args.control_grid_align_corners:
                cgs = x.new(cgs).view(1, 2)
                self.c_src.mul_(1 - 1 / cgs).add_(1 - 0.5 / cgs)
        super().build(x)

    def create_default_params(self, x):
        return dict(offsets=x.new_zeros((x.shape[0], *self.c_src.shape)))

    def forward(self, inputs):
        offsets = self.offsets
        if self.args.center_offsets:
            offsets.sub_(offsets.mean(list(range(1, len(offsets.shape))), keepdim=True))

        c_src = self.c_src.unsqueeze(0).expand_as(offsets)
        c_dst = c_src + offsets
        if self.args.swap_src_dst:
            c_src, c_dst = c_dst, c_src

        grid = (vmf.tps_grid_from_points if self.args.forward else
                vmf.backward_tps_grid_from_points)(c_src, c_dst, size=inputs[0].shape)
        return _warp_tuple(inputs, grid, **self.warp_args)

    @torch.no_grad()
    def inverse_module(self):
        inv = copy.copy(self)
        inv.args = type(self.args)({**self.args, "forward": not self.args.forward,
                                    "swap_src_dst": not self.args.swap_src_dst})
        return inv


BackwardTPSWarp = partial(TPSWarp, forward=False)


def cutmix_pairs_transform(x, mask):
    x_p = x.clone()
    if len(x) % 2 != 0:
        x_p[:-1:2][mask[:-1]], x_p[1::2][mask[:-1]] = x[1::2][mask[:-1]], x[:-1:2][mask[:-1]]
        x_p[-1:][mask[-1:]] = x_p[0:1][mask[-1:]]
    else:
        x_p[::2][mask], x_p[1::2][mask] = x[1::2][mask], x[::2][mask]
    return x_p


def cutmix_roll_transform(x, mask):
    x_p = x.clone()
    x_p[mask] = torch.roll(x, 1)[mask]
    return x_p


class CutMix(PertModel):
    domain = dt.ArraySpatial2D

    def __init__(self, mask_gen, combination: T.Literal['pairs', 'roll'] = 'pairs'):
        check_argument_types()
        super().__init__()
        self.mask_gen = mask_gen
        self.combination = combination

    def build(self, inputs):
        x = next(x for x in inputs if isinstance(x, dt.Image))

        if self.combination == 'pairs' and len(x) % 2 != 0:
            warn("There has to be an even number of examples for the 'pairs' combination mode.")
        n = (len(x) + 1) // 2 if self.combination == 'pairs' else len(x)
        self.register_buffer('mask', self.mask_gen(n, tuple(x.shape[-2:]), device=x.device))
        super().build(x)

    def _adapt_mask(self, x, mask):
        if len(x.shape) == 4:
            mask = mask.view(len(mask), 1, *x.shape[2:])
        return mask.expand(-1, *x.shape[1:])

    def forward(self, inputs):
        transform = cutmix_pairs_transform if self.combination == 'pairs' else cutmix_roll_transform
        return [
            transform(x, self._adapt_mask(x, self.mask)) if isinstance(x, dt.ArraySpatial2D) else
            x
            for x in inputs]
