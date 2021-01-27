import torch

import vidlu.training.extensions as te
from vidlu.modules import get_submodule
from vidlu.utils.collections import NameDict
from vidlu.utils.func import params, partial, Required


# Optimizer maker


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

    def __init__(self, optimizer_f, params, ignore_remaining_params=False, **kwargs):
        self.optimizer_f, self.params, self.kwargs = optimizer_f, params, kwargs
        self.ignore_remaining_params = ignore_remaining_params

    def __call__(self, model):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"The model argument should be a nn.Module, not {type(model)}.")
        params = [{
            **d, 'params': tuple(get_submodule(model, d['params']).parameters())}
            for d in self.params]
        params_lump = set(p for d in params for p in d['params'])
        remaining_params = () if self.ignore_remaining_params \
            else tuple(p for p in model.parameters() if p not in params_lump)
        return self.optimizer_f([{'params': remaining_params}] + params, **self.kwargs)


# Trainer config


class TrainerConfig(NameDict):
    def __init__(self, *args, **kwargs):
        ext_args = []  # extension factories are concatenated in order of appearance
        all_kwargs = {}  # other kwargs are updated with potential overriding
        for x in args:
            if isinstance(x, TrainerConfig):
                d = dict(**x)
                ext_args.extend(d.pop('extension_fs', ()))
                all_kwargs.update(d)
            elif issubclass(x.func if isinstance(x, partial) else x, te.TrainerExtension):
                ext_args.append(x)
            else:
                raise ValueError(f"Invalid argument type: {type(x).__name__}.")
        ext = tuple(kwargs.pop('extension_fs', ())) + tuple(ext_args)
        all_kwargs.update(kwargs)
        super().__init__(**all_kwargs, extension_fs=ext)

    def normalized(self):
        """Creates an equivalent TrainerConfig where arguments for extensions
         are bound to corresponding extension factories and removed from the
         main namespace.

        A normalized TrainerConfig can be given to the Trainer constructor.

        Example:
            >>> from vidlu.training.trainers import Trainer
            >>> tc: TrainerConfig(...)
            >>> trainer = Trainer(**tc.normalized())
        """
        result = TrainerConfig(**self)
        arg_name_to_ext = dict()
        ext = []
        for ext_f in result.extension_fs:
            names = tuple(params(ext_f).keys())
            values = [result.pop(name, Required) for name in names]
            args = {k: v for k, v in zip(names, values) if v is not Required}
            ext.append(partial(ext_f, **args) if len(args) > 0 else ext_f)
            for name in names:
                if name in arg_name_to_ext:
                    raise RuntimeError(f'Multiple extension factories have a parameter "{name}".')
                arg_name_to_ext[name] = ext_f
        result.extension_fs = ext
        return result
