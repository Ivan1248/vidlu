import functools

import vidlu.training.extensions as te
from vidlu.utils.collections import NameDict
from vidlu.utils.func import params, partial, Required, ArgTree, tree_partial


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
        ext = []
        used_keys = set()
        for ext_f in result.extension_fs:
            names = tuple(params(ext_f).keys())
            args = {name: result[name] for name in names if name in result}
            used_keys.update(args.keys())
            ext.append(partial(ext_f, **args) if len(args) > 0 else ext_f)

        for k in used_keys:
            del result[k]

        result.extension_fs = ext
        return result
