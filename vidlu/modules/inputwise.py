import typing as T
from functools import partial
import inspect
from vidlu.torch_utils import round_float_to_int
import warnings

import torch
import torch.nn.functional as F

import vidlu.modules.elements as E
import vidlu.modules.functional as vmf
from vidlu.modules.utils import sole_tuple_to_varargs
from vidlu.utils.collections import NameDict


class BatchParameter(torch.nn.Parameter):
    r"""A kind of Tensor that is to be considered a batch of parameters, i.e.
    each input example has its own parameter.
    Arguments:
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
               else partial(torch.full, fill_value=factory_or_value))
    return factory(x.shape, device=x.device)


def _complete_shape(shape_tail, input_shape):
    return input_shape[:-len(shape_tail)] + tuple(
        b if a is None else a for a, b in zip(shape_tail, input_shape[-len(shape_tail):]))


class PerturbationModel(E.Module):
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

    def build(self, x):
        # dummy_x has properties like x, but takes up almost no memory
        self.dummy_x = x.new_zeros(()).expand(x.shape)
        default_params = self.create_default_params(x)
        assert set(default_params) == set(self.param_defaults)
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
            lambda m: getpd(m) if isinstance(m, PerturbationModel) else iter(()),
            prefix=prefix, recurse=recurse)

    def reset_parameters(self):
        for (name, data), (name_, def_value) in zip(self.named_parameters(),
                                                    self.named_default_parameters(True)):
            assert name == name_
            data.set_(def_value)

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        warnings.warn(f"ensure_output_within_bounds is called"
                      + f" but not implemented for {type(self).__name__}")


def default_parameters(pert_model, full_size: bool, recurse=True):
    return PerturbationModel.default_parameters(pert_model, full_size, recurse=recurse)


def named_default_parameters(pert_model, full_size: bool, recurse=True):
    return PerturbationModel.named_default_parameters(pert_model, full_size, recurse=recurse)


def reset_parameters(pert_model):
    PerturbationModel.reset_parameters(pert_model)


class PerturbationModelWrapper(PerturbationModel):
    def __init__(self, module, forward_arg_count=None):
        super().__init__(forward_arg_count=forward_arg_count)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class SimplePerturbationModel(PerturbationModel):
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

    # def difference_from_default_params(self):
    #     return {k: getattr(self, k) - v['value']
    #             for k, v in self.param_defaults.items()}


class AlterGamma(SimplePerturbationModel):
    param_defaults = dict(gamma=dict(value=1., bounds=[0, 500]))
    eps = 1e-8

    def forward(self, x):
        return x.mul_(1 - 2 * self.eps).add_(self.eps).pow(self.gamma)


class AlterLogGamma(SimplePerturbationModel):
    # Gradients are more stable than for AlterGamma
    param_defaults = dict(log_gamma=dict(value=0., bounds=[-6, 6]))

    def forward(self, x):
        return x.pow(self.log_gamma.exp())


class AlterContrast(SimplePerturbationModel):
    param_defaults = dict(contrast=dict(value=1., bounds=[0, 500]))

    def forward(self, x):
        return (x - 0.5).mul_(self.contrast).add_(0.5)


class Additive(SimplePerturbationModel):
    param_defaults = dict(addend=dict(value=0., bounds=[-1, 1]))

    def forward(self, x):
        return x + self.addend

    def ensure_output_within_bounds(self, x, bounds, computed_output=None):
        self.addend.add_(x).clamp_(*bounds).sub_(x)


class Multiplicative(SimplePerturbationModel):
    param_defaults = dict(factor=dict(value=1., bounds=[0, 500]))

    def forward(self, x):
        return self.factor * x


class Whiten(SimplePerturbationModel):
    """Interpolates pixel values between the original ones and 1."""
    param_defaults = dict(weight=dict(value=0., bounds=[0, 1]))

    def forward(self, x):
        return (1 - x).mul_(self.weight).add_(x)


class Warp(PerturbationModel):
    param_defaults = dict(flow=dict(value=0., bounds=[0, 1]))

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=True):
        super().__init__()
        self.args = dict(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    def create_default_params(self, x):
        return dict(flow=x.new_zeros((x.shape[0], 2, *x.shape[2:])))

    def forward(self, x):
        return vmf.warp(x, self.flow, **self.args)


def _grid_sample(x, grid, y=None, interpolation_mode='bilinear', padding_mode='zeros',
                 align_corners=True, label_interpolation_mode='nearest',
                 label_padding_mode=-1):
    pm, lpm = ['zeros' if x == 0 else x for x in [padding_mode, label_padding_mode]]

    if isinstance(pm, str):
        x_p = F.grid_sample(x, grid, mode=interpolation_mode, padding_mode=pm,
                            align_corners=align_corners).squeeze_(1)
    else:
        x_p = F.grid_sample(x - pm, grid, mode=interpolation_mode, padding_mode='zeros',
                            align_corners=align_corners).add_(pm)

    if y is None:
        return x_p
    elif y.dim() < 3:
        y_p = y
    else:
        y_p = (y.to(grid.dtype) if 'float' not in f'{y.dtype}' else y)
        no_channel_dim = y.dim() == 3
        if no_channel_dim:
            y_p = y_p[:, None, ...]
        if isinstance(lpm, str):
            y_p = F.grid_sample(y_p, grid, mode=label_interpolation_mode, padding_mode=lpm,
                                align_corners=align_corners).squeeze_(1)
        else:
            y_p = F.grid_sample(y_p - lpm, grid, mode=label_interpolation_mode,
                                padding_mode='zeros', align_corners=align_corners).add_(lpm)
        if no_channel_dim:
            y_p.squeeze_(1)
        if y_p.dtype is not y.dtype:
            y_p = round_float_to_int(y_p, y.dtype)
    return x_p, y_p


class MorsicTPSWarp(PerturbationModel):
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
        return _grid_sample(x, y=y, grid=grid,
                            **{k: self.args[k]
                               for k in ['interpolation_mode', 'padding_mode',
                                         'label_interpolation_mode', 'label_padding_mode']})


class BackwardTPSWarp(PerturbationModel):
    param_defaults = dict(offsets=dict(value=0., bounds=[-0.2, 0.2]))

    def __init__(self, control_grid_shape=(2, 2), control_grid_align_corners=False,
                 align_corners=True, padding_mode='zeros', interpolation_mode='bilinear',
                 label_interpolation_mode='nearest', label_padding_mode=-1):
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
        grid = vmf.backward_tps_grid_from_points(c_src, c_src + self.offsets, size=x.shape)
        return _grid_sample(x, y=y, grid=grid,
                            **{k: self.args[k]
                               for k in ['interpolation_mode', 'padding_mode',
                                         'label_interpolation_mode', 'label_padding_mode']})
