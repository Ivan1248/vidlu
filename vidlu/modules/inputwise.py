import typing as T
from vidlu.utils.func import partial
import inspect
import copy
import warnings
import contextlib as ctx

import torch
import torch.nn.functional as F
from numpy import s_
from vidlu.torch_utils import round_float_to_int
import vidlu.modules.elements as E
import vidlu.modules.functional as vmf
from vidlu.modules.utils import sole_tuple_to_varargs
import vidlu.modules.components as vmc
import vidlu.ops.image as voi


class BatchParameter(torch.nn.Parameter):
    r"""A kind of Tensor that is to be considered a batch of parameters, i.e.
    each input example has its own parameter.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def __repr__(self):
        return 'BatchParameter:\n' + repr(self.data)

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


class PertModelBase(E.Module):
    param_defaults = dict()

    def __init__(self, forward_arg_count=None):
        super().__init__()
        if forward_arg_count is None:
            self.forward_arg_count = 0
            unlimited_param_kinds = (
                inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            for p in inspect.signature(self.forward).parameters.values():
                self.forward_arg_count += 1
                if p.kind in unlimited_param_kinds:
                    self.forward_arg_count = -1
                    break
        else:
            self.forward_arg_count = forward_arg_count
        # self.init = init

    # def initialize(self, input=None):  # TODO: make initialize?
    #     if input is not None:
    #         self(input)
    #     self.init(self, input)

    def build(self, x):
        # dummy_x has properties like x, but takes up almost no memory
        self.dummy_x = x.new_zeros(()).expand(x.shape)
        default_params = self.create_default_params(x)
        if (dp := set(default_params)) != (pd := set(self.param_defaults)):
            raise TypeError(f"The names of parameters returned by `create_default_params` ({dp}) no"
                            f" not match the expected names per `param_defaults` ({pd}).")
        for k, v in default_params.items():
            setattr(self, k, BatchParameter(v, requires_grad=True))

    def __call__(self, *args, **kwargs):
        args = sole_tuple_to_varargs(args)
        n = self.forward_arg_count
        if len(args) + len(kwargs) == 1 or n == -1:  # everything fits
            result = super().__call__(*args, **kwargs)
        elif len(args) >= n:  # args don't fit
            result = (super().__call__(args[0]),) if n == 1 else super().__call__(*args[:n])
            result += (*args[n:], *tuple(kwargs.values()))
        else:  # kwargs don't fit
            r = n - len(args)
            kw, unchanged = dict(tuple(kwargs.items())[:r]), tuple(kwargs.values())[r:]
            result = super().__call__(*args, **kw) + unchanged
        return result

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
            lambda m: getpd(m) if isinstance(m, PertModelBase) else iter(()),
            prefix=prefix, recurse=recurse)

    def reset_parameters(self):
        for (name, data), (name_, def_value) in zip(self.named_parameters(),
                                                    self.named_default_parameters(True)):
            assert name == name_
            data.set_(def_value if not callable(def_value)
                                   and not isinstance(def_value, torch.Tensor) else
                      def_value())

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        warnings.warn(f"ensure_output_within_bounds is called"
                      + f" but not implemented for {type(self).__name__}")


def default_parameters(pert_model, full_size: bool, recurse=True):
    return PertModelBase.default_parameters(pert_model, full_size, recurse=recurse)


def named_default_parameters(pert_model, full_size: bool, recurse=True):
    return PertModelBase.named_default_parameters(pert_model, full_size, recurse=recurse)


def reset_parameters(pert_model):
    PertModelBase.reset_parameters(pert_model)


# TODO: redesign perturbation models using vidlu.modules.Parallel (remove forward_arg_count)

def _blend(x1, x2, ratio):
    return (ratio * x1).add_((1.0 - ratio) * x2)


class PertModel(PertModelBase):
    def __init__(self, module, forward_arg_count=None):
        super().__init__(forward_arg_count=forward_arg_count)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class EquivariantPertModel(PertModelBase):
    param_defaults = dict()

    def __init__(self, equivariant_dims: T.Sequence):
        super().__init__()
        self.equivariant_dims = equivariant_dims

    def create_default_params(self, x):
        shape = list(x.shape)
        for d in self.equivariant_dims:
            shape[d if d >= 0 else len(x.shape) - d] = 1
        dummy = x.new_zeros(()).expand(shape)  # contains shape, dtype, and device
        return {k: _get_param(dummy, self.param_defaults[k]['value'])
                for k, v in self.param_defaults.items()}


class SliceEquivariantPertModel(PertModelBase):
    """Can modify only a slice, e.g. a channel."""
    param_defaults = dict()

    def __init__(self, equivariant_dims: T.Sequence, slice=None):  # slice=np.s_[...] can be used
        super().__init__()
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


class AlterGamma(EquivariantPertModel):
    param_defaults = dict(gamma=dict(value=1., bounds=[0, 500]))
    eps = 1e-8

    def forward(self, x):
        return x.mul_(1 - 2 * self.eps).add_(self.eps).pow(self.gamma)


class AlterLogGamma(EquivariantPertModel):
    # Gradients are more stable than for AlterGamma
    param_defaults = dict(log_gamma=dict(value=0., bounds=[-6, 6]))

    def forward(self, x):
        return x.pow(self.log_gamma.exp())


def _get_center(center_arg, x, equivariant_dims):
    return (x.mean(equivariant_dims, keepdim=True) if center_arg == 'mean' else
            center_arg(x) if callable(center_arg) else center_arg)


class Contrast(EquivariantPertModel):
    """Linearly interpolates between inputs and centers.

    The interpolation factor (contrast) can be any positive number.

    To make it work like adjust_contrast in Torchvision, center should be set to
    the average luma value across the whole image according to CCIR 601:

    >>> from vidlu.ops.image import rgb_to_luma
    >>> center = lambda x: rgb_to_luma(x, 601).unsquezze(-3).mean((-2, -1), keepdim=True)
    >>> torchvision_contrast = Contrast(cente[r=center)
    """
    param_defaults = dict(factor=dict(value=1., bounds=[0, float('inf')]))

    def __init__(self, equivariant_dims=(-2, -1),
                 center: T.Union[float, T.Callable, T.Literal['mean']] = 0.5):
        super().__init__(equivariant_dims)
        self.center = center

    def forward(self, x):
        center = _get_center(self.center, x, equivariant_dims=self.equivariant_dims)
        return _blend(x, center, self.factor)


class TorchvisionSaturation(Contrast):
    param_defaults = dict(factor=dict(value=1., bounds=[0, float('inf')]))

    def __init__(self, greyscale_f=partial(voi.rgb_to_luma, rev=601)):
        super().__init__(equivariant_dims=())
        self.greyscale_f = greyscale_f

    def forward(self, x):
        grey = self.greyscale_f(x)
        if len(grey.shape) < len(x.shape):
            grey = grey.unsqueeze(-3)
        return _blend(x, grey, self.factor)


class Add(SliceEquivariantPertModel):
    param_defaults = dict(addend=dict(value=0., bounds=[-1, 1]))

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
        if self.slice is None:
            return x % self.factor
        y = x.clone()
        y[self.slice] %= self.period
        return y


class Whiten(EquivariantPertModel):
    """Interpolates pixel values between the original ones and 1."""
    param_defaults = dict(weight=dict(value=0., bounds=[0, 1]))

    def forward(self, x):
        return ((1 - self.weight) * x).add_(self.weight)


class AlterColor(PertModel):
    """Applies a transformation similar to ColorJitter in Torchvision.
    """
    param_defaults = dict(
        order=dict(value=lambda **k: torch.arange(len(k['x']), len(k['self'].module)),
                   bounds=[0, 1]))

    def __init__(self, brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                 hue=(-0.5, 0.5)):
        modules = dict(brightness=Multiply(equivariant_dims=(-1, -2)),
                       contrast=Contrast(center='mean'),
                       saturation=TorchvisionSaturation(),
                       hue=E.Seq(to_hsv=PertModel(voi.rgb_to_hsv, forward_arg_count=1),
                                 add_h=Add((2, 3), slice=s_[:, 0:, ...]),  # no need for modulo
                                 to_rgb=PertModel(voi.hsv_to_rgb, forward_arg_count=1)))
        super().__init__(
            E.ModuleTable({k: v for k, v in modules.items() if locals()[k] is not None}))
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

    def forward(self, x):
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


class Warp(PertModelBase):
    param_defaults = dict(flow=dict(value=0., bounds=[-1, 1]))

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def create_default_params(self, x):
        return dict(flow=x.new_zeros((x.shape[0], 2, *x.shape[2:])))

    def forward(self, x):
        return vmf.warp(x, self.flow, **self.args)


class SmoothWarp(PertModelBase):
    param_defaults = dict(unsmoothed_flow=dict(value=0., bounds=[-100, 100]))

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True,
                 smooth_f=partial(vmc.GaussianFilter2D, sigma=3)):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        self.smooth = smooth_f()

    def create_default_params(self, x):
        return dict(unsmoothed_flow=x.new_zeros((x.shape[0], 2, *x.shape[2:])))

    def forward(self, x):
        flow = self.smooth(self.unsmoothed_flow)
        return vmf.warp(x, flow, **self.args)


def _grid_sample_p(x, grid, mode, padding_mode, align_corners):
    pm = 'zeros' if padding_mode == 0 else padding_mode
    if isinstance(pm, str):
        return F.grid_sample(x, grid, mode=mode, padding_mode=pm,
                             align_corners=align_corners).squeeze_(1)
    else:
        return F.grid_sample(x - pm, grid, mode=mode, padding_mode='zeros',
                             align_corners=align_corners).squeeze_(1).add_(pm)


def _forward_warp(x, grid, mode, padding_mode, align_corners):
    # https://github.com/sniklaus/softmax-splatting
    from vidlu.libs.softmax_splatting import softsplat
    pv = 0 if padding_mode in (0, 0., "zeros") else padding_mode
    warnings.warn("align_corners and mode not used in _forward_warp")
    H, W = grid.shape[-3:-1]
    base_grid = vmf.uniform_grid_2d((H, W), low=-1., high=1., device=grid.device, dtype=grid.dtype)
    offsets = grid - base_grid
    flow = offsets.permute(0, 3, 1, 2).mul_(offsets.new([W / 2, H / 2]).view(1, 2, 1, 1))
    result = softsplat.FunctionSoftsplat(x if pv == 0 else x - pv, flow.contiguous(),
                                         tenMetric=None, strType="average")
    return result if pv == 0 else result.add_(pv)


def _warp(x, grid, y=None, interpolation_mode='bilinear', padding_mode='zeros',
          align_corners=True, label_interpolation_mode='nearest',
          label_padding_mode=-1, forward=False):
    pm, lpm = ['zeros' if m == 0 else m for m in [padding_mode, label_padding_mode]]
    _warp_func = _forward_warp if forward else _grid_sample_p

    x_p = None if x is None else _warp_func(x, grid, mode=interpolation_mode, padding_mode=pm,
                                            align_corners=align_corners)
    if y is None:
        return x_p
    if y.dim() < 3:
        y_p = y
    else:
        y_p = (y if 'float' in f'{y.dtype}' else y.to(grid.dtype))
        single_channel = y.dim() == 3
        y_p = _warp_func(y_p[:, None, ...] if single_channel else y_p, grid,
                         mode=label_interpolation_mode, padding_mode=lpm,
                         align_corners=align_corners)
        if single_channel:
            y_p.squeeze_(1)
        if y_p.dtype is not y.dtype:
            y_p = round_float_to_int(y_p, y.dtype)
    return x_p, y_p


class MorsicTPSWarp(PertModelBase):
    # directly uses theta, skipping control points.
    # Use BackwardTPSWarp instead.
    param_defaults = dict(theta=dict(value=0., bounds=[-0.5, 0.5]))

    def __init__(self, grid_shape=(2, 2), align_corners=True, padding_mode='zeros',
                 interpolation_mode='bilinear', label_interpolation_mode='nearest',
                 label_padding_mode=-1):
        super().__init__()
        self.store_args()

    def build(self, x):
        k = dict(device=x.device, dtype=x.dtype)
        self.c_dst = vmf.uniform_grid_2d(self.args.grid_shape, **k).view(-1, 2)
        super().build(x)

    def create_default_params(self, x):
        return dict(theta=x.new_zeros((x.shape[0], self.c_dst.shape[0] + 2, 2)).squeeze(-1))

    def forward(self, x, y=None):
        grid = vmf.tps_grid(self.theta, self.c_dst, x.shape)
        return _warp(x, y=y, grid=grid,
                     **{k: self.args[k]
                        for k in ['interpolation_mode', 'padding_mode',
                                  'label_interpolation_mode', 'label_padding_mode']})


class TPSWarp(PertModelBase):
    param_defaults = dict(offsets=dict(value=0., bounds=[-0.2, 0.2]))

    def __init__(self, *, forward: bool, control_grid_shape=(2, 2),
                 control_grid_align_corners=False, align_corners=True, padding_mode='zeros',
                 interpolation_mode='bilinear', label_interpolation_mode='nearest',
                 label_padding_mode=-1, swap_src_dst=False):
        super().__init__()
        self.store_args()

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

    def forward(self, x, y=None):
        c_src = self.c_src.unsqueeze(0).expand_as(self.offsets)
        c_dst = c_src + self.offsets
        if self.args.swap_src_dst:
            c_src, c_dst = c_dst, c_src

        grid = (vmf.tps_grid_from_points if self.args.forward else
                vmf.backward_tps_grid_from_points)(c_src, c_dst, size=x.shape)
        return _warp(x, y=y, grid=grid,
                     **{k: self.args[k]
                        for k in ['interpolation_mode', 'padding_mode',
                                  'label_interpolation_mode', 'label_padding_mode',
                                  'forward']})

    @torch.no_grad()
    def inverse_module(self):
        inv = copy.copy(self)
        inv.args = type(self.args)({**self.args, "forward": not self.args.forward,
                                    "swap_src_dst": not self.args.swap_src_dst})
        return inv


BackwardTPSWarp = partial(TPSWarp, forward=False)
