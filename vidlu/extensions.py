"""
Names of extension packages have to start with "vidlu_". This corresponds to the *naming convention*
approach described here:
https://packaging.python.org/guides/creating-and-discovering-plugins/#using-naming-convention
The prefix is removed when accessing the extension via `vidlu.extension.ext_name` if the name of the
package is `vidlu_ext_name`.

For non-installed packages, add the path of the directory containing extensions to `PYTHONPATH`.
```PYTHONPATH="${PYTHONPATH}:/my/plugins/path"```
"""

import importlib
import pkgutil

EXT_PREFIX = "vidlu_"

globals().update({name[len(EXT_PREFIX):]: importlib.import_module(name)
                  for finder, name, ispkg in pkgutil.iter_modules()
                  if name.startswith(EXT_PREFIX)})
del importlib, pkgutil


def __getattr__(name):
    if name.startswith(EXT_PREFIX) and name not in globals():
        msg = f'Extensions should be accessed without the prefix "{EXT_PREFIX}".'
    else:
        msg = f'There is no extension with name "{name}" (package name "{EXT_PREFIX + name}").'
    raise AttributeError(msg)
