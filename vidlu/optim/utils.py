import typing as T

import torch

from vidlu.modules import get_submodule, Module


class OptimizerMaker:
    """A type for storing all optimizer information without depending on a
    model instance by storing module names instead of parameters.

    Calling an object of this type with a model creates an optimizer instance.

    Args:
        optimizer_f: PyTorch optimizer factory (or constructor).
        params (List[Mapping]): A list of dictionaries in the same format as
            for PyTorch optimizers except for module names instead of
            parameters.
        ignore_remaining_params: Specifies whether unlisted parameters should be
            ignored instead of being optimized.
        **kwargs: Keyword arguments for the optimizer.
    """

    def __init__(self, optimizer_f, params: T.Sequence[T.Mapping], *,
                 ignore_remaining_params: bool = False,
                 **kwargs):
        self.optimizer_f, self.params, self.kwargs = optimizer_f, params, kwargs
        self.ignore_remaining_params = ignore_remaining_params

    def __call__(self, model):
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"model should be a nn.Module, not {type(model)}.")
        if isinstance(model, Module) and not model.is_built():
            raise RuntimeError(f"model should be initialized.")

        def get_params(names_or_funcs):
            if isinstance(names_or_funcs, str) or callable(names_or_funcs):
                names_or_funcs = [names_or_funcs]
            return [p for nf in names_or_funcs for p in (
                get_submodule(model, nf).parameters() if isinstance(nf, str) else nf(model))]

        params = [{**d, 'params': get_params(d['params'])} for d in self.params]
        params_lump = set(p for d in params for p in d['params'])
        remaining_params = () if self.ignore_remaining_params \
            else tuple(p for p in model.parameters() if p not in params_lump)
        return self.optimizer_f([{'params': remaining_params}] + params, **self.kwargs)

    def __repr__(self):
        return f"OptimizerMaker(optimizer_f={repr(self.optimizer_f)}, params={repr(self.params)}," \
               + f" kwargs={repr(self.kwargs)}," \
               + f"ignore_remaining_params={self.ignore_remaining_params})"
