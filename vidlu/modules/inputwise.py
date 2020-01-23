from typing import Sequence

import torch

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


def _get_param(shape, factory_or_value):
    return BatchParameter(factory_or_value(shape, requires_grad=True) if callable(factory_or_value)
                          else torch.full(shape, factory_or_value, requires_grad=True))


def _complete_shape(shape_tail, input_shape):
    return input_shape[:-len(shape_tail)] + tuple(
        b if a is None else a for a, b in zip(shape_tail, input_shape[-len(shape_tail):]))


class PerturbationModel(E.Module):
    param_defaults = dict()

    def build(self, x):
        self.input_shape = x.shape
        for k, v in self.create_default_params(x.shape).items():  # TODO: devicd
            setattr(self, k, v)

    def create_default_params(self, input_shape):
        raise NotImplementedError()

    def default_parameters(self, minimum_shape=False, recurse=True):
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
        for _, param in self.named_default_parameters(minimum_shape=minimum_shape, recurse=recurse):
            yield param

    def named_default_parameters(self, minimum_shape=False, prefix='', recurse=True):
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
        getpd = ((lambda m: ((k, v['value']) for k, v in m.param_defaults.items())) if minimum_shape
                 else lambda m: m.create_default_params(self.input_shape).items())
        return self._named_members(
            lambda m: getpd(m) if isinstance(m, PerturbationModel) else iter(()),
            prefix=prefix, recurse=recurse)


class SimplePerturbationModel(PerturbationModel):
    param_defaults = dict()

    def __init__(self, equivariant_dims: Sequence):
        super().__init__()
        self.equivariant_dims = equivariant_dims

    def create_default_params(self, input_shape):
        shape = list(input_shape)
        for d in self.args.equivariant_dims:
            shape[d if d >= 0 else len(input_shape) - d] = 1
        return {k: _get_param(shape, self.param_defaults[k]['value'])
                for k, v in self.param_defaults.items()}

    # def difference_from_default_params(self):
    #     return {k: getattr(self, k) - v['value']
    #             for k, v in self.param_defaults.items()}


class AlterGamma(SimplePerturbationModel):
    param_defaults = dict(gamma=dict(value=1., bounds=[0, 500]))

    def forward(self, x):
        return x.pow(self.gamma)


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
    param_defaults = dict(factor=dict(value=0., bounds=[0, 1]))

    def create_default_params(self, input_shape):
        return dict(flow=torch.zeros((input_shape[0], 2, *input_shape[2:]), requires_grad=True))

    def forward(self, x):
        return vmf.warp(x, self.flow)
