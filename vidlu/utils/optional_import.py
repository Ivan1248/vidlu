import typing
import warnings
import importlib
import sys
import types


def set_module_substitute(name, substitute):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        warnings.warn(f"No module named '{name}'. A mock module will be used instead.")
        # TODO: create a replacement module with optional_package_mocks and put it into sys.modules
        if isinstance(substitute, typing.Mapping):
            substitute_module = types.ModuleType(name)
            substitute_module.__dict__.update(substitute)
        else:
            substitute_module = substitute
        sys.modules[name] = substitute_module
