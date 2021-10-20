"""
Names of extension packages have to start with "vidlu_". This corresponds to the *naming convention*
approach described here:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-naming-convention
The prefix is removed when accessing the extension via `vidlu.extension.ext_name` if the name of the
package is `vidlu_ext_name`.

For non-installed packages, add the path of the directory containing extensions to `PYTHONPATH`.
```PYTHONPATH="${PYTHONPATH}:/my/plugins/path"```
"""

import sys
import importlib
import importlib.util
import pkgutil
from functools import partial
import traceback

from vidlu.utils.collections import NameDict

EXT_PREFIX = "vidlu_"


def lazy_import(fullname):
    """Returns a module that is not imported until used.

    From https://stackoverflow.com/questions/42703908/how-do-i-use-importlib-lazyloader
    """
    try:
        return sys.modules[fullname]
    except KeyError:
        spec = importlib.util.find_spec(fullname)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        # Make module with proper locking and get it inserted into sys.modules.
        loader.exec_module(module)
        return module


class ExtensionDict(NameDict):
    def __getattr__(self, name):
        try:
            super().__getattr__(name)
        except AttributeError as e:
            if name.startswith(EXT_PREFIX):
                msg = f'Extensions should be accessed without the prefix "{EXT_PREFIX}".'
            else:
                msg = f'There is no extension with name "{name}" (package name "{EXT_PREFIX + name}").'
            raise AttributeError(msg)


class LazyObjectProxy:
    def __init__(self, factory):
        self._lazy_factory = factory
        self._lazy_obj = None

    def __load(self):
        if object.__getattribute__(self, "_lazy_obj") is None:
            self._lazy_obj = object.__getattribute__(self, "_lazy_factory")()

    def __getattribute__(self, item):
        LazyObjectProxy.__load(self)
        return getattr(object.__getattribute__(self, "_lazy_obj"), item)


class LazyModule(LazyObjectProxy):
    def __init__(self, name):
        super().__init__(partial(importlib.import_module, name))


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.write(f'Failed to load extension {name}.\n')
        return None

# A lazy object proxy is used to postpone import errors only when there are errors
extensions = ExtensionDict({
    name[len(EXT_PREFIX):]: m
    for finder, name, ispkg in pkgutil.iter_modules()
    if name.startswith(EXT_PREFIX) and (m := _try_import(name) or LazyModule(name)) is not None})
