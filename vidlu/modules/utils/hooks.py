import torch

from .utils import try_get_module_name_from_call_stack, extract_tensors
from vidlu.modules.tensor_extra import LogAbsDetJac


def check_no_inf_nan(module, input, output):
    """
    Example:
        >>> for m in module.modules:
        >>>      m.register_forward_hook(check_no_inf_nan)
    """
    if (inf := torch.isinf(output).any()) or (nan := torch.isnan(output).any()):
        module_name = try_get_module_name_from_call_stack(module)
        print(f"{module_name=}, {inf=}, {nan=}")
        breakpoint()


def check_propagates_log_abs_det_jac(module, input, output):
    if LogAbsDetJac.has(next(extract_tensors(input))):
        for i, y in enumerate(extract_tensors(output)):
            if not LogAbsDetJac.has(y):
                module_name = try_get_module_name_from_call_stack(module)
                raise RuntimeError(f"Output ({i}) of {module_name} ({type(module)}) has no"
                                   + f" `ln(abs(det(jacobian)))` despite input(s) having it.")
