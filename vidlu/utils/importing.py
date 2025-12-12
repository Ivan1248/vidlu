import importlib


# Import module ####################################################################################

def import_module_from_file(path):  # from morsic
    """Dynamically import a module from a given file path.

    Args:
        path (str): Path to the Python file to import.

    Returns:
        module: The imported module object.
    """
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_all(name: str, namespace, package: str | None = None):
    """Import all public attributes from a module into a given namespace.

    Args:
        name (str): Name of the module to import.
        namespace (dict): Dictionary representing the target namespace.
        package (str | None, optional): Package name for relative imports. Defaults to None.

    Returns:
        module: The imported module object.
    """
    module = importlib.import_module(name, package)

    # is there an __all__?  if so respect it
    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in module.__dict__ if not x.startswith("_")]

    # now drag them in
    namespace.update({k: getattr(module, k) for k in names})
    return module


def parse_aliased_imports_expression(imports: str, namespace=None):
    """Parse and execute a short imports expression.

    Expressions can import modules according to a list containing:
    - alias expressions (e.g., `alias=module_name`),
    - original module names (e.g., `module_name`).

    Args:
        imports (str): Short imports expression (comma-separated module names with optional aliases).
        namespace (dict, optional): Namespace to populate with imports. Defaults to None.

    Returns:
        dict: Updated namespace containing imported modules and attributes.
    """
    if namespace is None:
        namespace = dict()

    for se in imports.replace(' ', '').split(','):
        if se == '':
            continue
        if '=' in se:
            alias, name = se.split('=')
            namespace[alias] = importlib.import_module(name)
        else:
            import_all(se, namespace)

    return namespace
