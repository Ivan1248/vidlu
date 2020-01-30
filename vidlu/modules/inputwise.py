from typing import Sequence
from functools import partial
import inspect
from vidlu.utils.torch import round_float_to_int

import torch
import torch.nn.functional as F

import vidlu.modules.elements as E
import vidlu.modules.functional as vmf


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_arg_count = 0
        unlimited_param_kinds = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for p in inspect.signature(self.forward).parameters.values():
            self.forward_arg_count += 1
            if p.kind not in unlimited_param_kinds:
                self.forward_arg_count = -1
                break

    def build(self, x):
        # dummy_x has properties like x, but takes up almost no memory
        self.dummy_x = x.new_zeros(()).expand(x.shape)
        default_params = self.create_default_params(x)
        assert set(default_params) == set(self.param_defaults)
        for k, v in default_params.items():
            setattr(self, k, BatchParameter(v, requires_grad=True))

    def create_default_params(self, x):
        raise NotImplementedError(f"{type(self)}")

    def default_parameters(self, full_size, recurse=True):
        r"""Returns an iterator over default module parameters.

        Args:
            minimum_shape: If True, arrays (or scalars) of the minimum shape
                necessary to compute the difference of parameters to their
                default values are yielded. This can be useful for constraints
                or regularization. Otherwise, parameters like those used for
                initialization of the module are yielded.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only default parameters
                for this module.

        Yields:
            Parameter: module parameter
        """
        for _, param in self.named_default_parameters(full_size=full_size, recurse=recurse):
            yield param

    def named_default_parameters(self, full_size, prefix='', recurse=True):
        r"""Returns an iterator over just created default module parameters,
        yielding both the name of the default parameter and the parameter.

        Args:
            minimum_shape: If True, arrays (or scalars) of the minimum shape
                necessary to compute the difference of parameters to their
                default values are yielded. This can be useful for constraints
                or regularization. Otherwise, parameters like those used for
                initialization of the module are yielded.
            prefix (str): Prefix to prepend to all parameter names.
            recurse (bool): If True, then yields default parameters of this
                module and all submodules. Otherwise, yields only parameters of
                this module.

        Yields:
            (string, Tensor): Tuple containing the name and parameter
        """
        getpd = ((lambda m: m.create_default_params(self.dummy_x).items()) if full_size
                 else (lambda m: ((k, v['value']) for k, v in m.param_defaults.items())))
        return self._named_members(
            lambda m: getpd(m) if isinstance(m, PerturbationModel) else iter(()),
            prefix=prefix, recurse=recurse)

    def __call__(self, *args, **kwargs):
        n = self.forward_arg_count
        if len(args) + len(kwargs) == 1 or n == -1:
            return super().__call__(*args, **kwargs)
        elif len(args) <= n:
            return (*super().__call__(*args[:n]), *args[n:], *tuple(kwargs.values()))
        else:
            r = n - len(args)
            kw, unchanged = dict(tuple(kwargs.items())[:r]), tuple(kwargs.values())[r:]
            return super().__call__(*args, **kw) + unchanged


class SimplePerturbationModel(PerturbationModel):
    param_defaults = dict()

    def __init__(self, equivariant_dims: Sequence):
        super().__init__()
        self.equivariant_dims = equivariant_dims

    def create_default_params(self, x):
        shape = list(x.shape)
        for d in self.args.equivariant_dims:
            shape[d if d >= 0 else len(x.shape) - d] = 1
        dum = x.new_zeros(()).expand(shape)
        return {k: _get_param(dum, self.param_defaults[k]['value'])
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


class MorsicTPSWarp(PerturbationModel):
    param_defaults = dict(theta=dict(value=0., bounds=[-0.5, 0.5]))

    def __init__(self, grid_shape=(2, 2), align_corners=True, padding_mode='zeros',
                 label_padding_mode=-1):
        super().__init__()

    def build(self, x):
        k = dict(device=x.device, dtype=x.dtype)
        self.c_dst = vmf.uniform_grid_2d(self.args.grid_shape, **k).view(-1, 2)
        super().build(x)

    def create_default_params(self, x):
        return dict(theta=x.new_zeros((x.shape[0], self.c_dst.shape[0] + 2, 2)).squeeze(-1))

    def forward(self, x, y=None):
        grid = vmf.tps_grid(self.theta, self.c_dst, x.shape)
        k = dict(mode='bilinear', padding_mode='zeros') if 'float' in f"{x.dtype}" else dict(
            mode='')
        x_p = F.grid_sample(x, grid, mode='bilinear', padding_mode=self.args.padding_mode,
                            align_corners=self.args.align_corners)
        lpm = self.args.label_padding_mode
        if y is None:
            return x_p
        elif y.dim() < 3:
            y_p = y
        else:
            y_p = (y.to(grid.dtype) if 'float' not in f'{y.dtype}' else y)[:, None, ...]
            if not isinstance(lpm, str) and lpm != 0:
                y_p = F.grid_sample(y_p - lpm, grid, mode='nearest',
                                    padding_mode='zeros',
                                    align_corners=self.args.align_corners).add_(lpm)
            else:
                y_p = F.grid_sample(y_p, grid, mode='nearest', padding_mode=lpm,
                                    align_corners=self.args.align_corners).squeeze_(1)
            y_p.squeeze_(1)
            if y_p.dtype is not y.dtype:
                y_p = round_float_to_int(y_p, y.dtype)
        return x_p, y_p
