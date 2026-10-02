"""
Names of extension packages have to be prefixed with "vidlu_". This corresponds to the *naming
convention* approach described here:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-naming-convention
The prefix is removed when accessing the extension, For example, the package with the name
`vidlu_ext_name` is accessible through `vidlu.extensions.extensions.ext_name`.

For non-installed packages, add the path of the directory containing extensions to `PYTHONPATH`.
```PYTHONPATH="${PYTHONPATH}:/my/vidlu/extensions/path"```
"""

import importlib
import pkgutil
from functools import partial

from vidlu.utils.collections import NameDict

EXT_PREFIX = "vidlu_"


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


# Extensions are imported on first use, so an extension that fails to import breaks only the
# runs that use it, and importing an extension cannot close a cycle with a partially imported
# `vidlu` module.
extensions = ExtensionDict({
    name[len(EXT_PREFIX):]: LazyModule(name)
    for finder, name, ispkg in pkgutil.iter_modules()
    if name.startswith(EXT_PREFIX)})
