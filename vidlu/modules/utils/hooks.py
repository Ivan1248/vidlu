import torch

from .utils import try_get_module_name_from_call_stack


def check_inf_nan_hook(module, input, output):
    """
    Example:
        >>> for m in module.modules:
        >>>      m.register_forward_hook(check_inf_nan_forward_hook)
    """
    if (inf := torch.isinf(output).any()) or (nan := torch.isnan(output).any()):
        module_name = try_get_module_name_from_call_stack(module)
        print(f"{module_name=}, {inf=}, {nan=}")
        breakpoint()
