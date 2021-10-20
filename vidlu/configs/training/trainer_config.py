import torch

import vidlu.training.extensions as te
from vidlu.modules import get_submodule
from vidlu.utils.collections import NameDict
from vidlu.utils.func import params, partial, Required, ArgTree, tree_partial
import functools


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


# Trainer config


def distribute_argtree_to_funcs(base_funcs, argtree_args):  # not used
    func_args_pairs = [(f, params(f)) for f in base_funcs]
    argtrees = [ArgTree() for _ in func_args_pairs]

    for k, argtree in argtree_args.items():
        found = False
        for i, (f, args) in func_args_pairs:
            if k in args:
                if found:
                    raise RuntimeError(f"Multiple functions have a parameter {k}"
                                       + f" and it cannot be unambigously bound.")
                argtrees[i][k] = argtree
                found = True
        if not found:
            raise RuntimeError(f"No function has a parameter {k}.")
    return [tree_partial(f, argtree) if len(argtree) > 0 else f
            for f, argtree in zip(base_funcs, argtrees)]


class TrainerConfig(NameDict):
    def __init__(self, *args, **kwargs):
        ext_args = []  # extension factories are concatenated in order of appearance
        all_kwargs = {}  # other kwargs are updated with potential overriding
        for x in args:
            if isinstance(x, TrainerConfig):
                d = dict(**x)
                ext_args.extend(d.pop('extension_fs', ()))
                all_kwargs.update(d)
            elif (isinstance(t := x.func if isinstance(x, functools.partial) else x, type)
                  and issubclass(t, te.TrainerExtension)):
                ext_args.append(x)
            else:
                raise ValueError(f"Invalid argument type: {type(x).__name__}.")
        ext = tuple(kwargs.pop('extension_fs', ())) + tuple(ext_args)
        all_kwargs.update(kwargs)
        # argtree_kwargs = {k: v for k, v in all_kwargs.items() if isinstance(v, ArgTree)}
        # normal_kwargs = {k: v for k, v in all_kwargs if k not in argtree_kwargs.items()}
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
