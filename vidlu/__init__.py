from .version import __version__
from .utils import optional_import

optional_package_mocks = dict(
    typeguard=dict(typechecked=lambda f: f))
for name, attributes in optional_package_mocks.items():
    optional_import.set_module_substitute(name, attributes)
