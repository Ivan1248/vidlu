import torch
import functools
import contextlib

from .utils import try_get_module_name_from_call_stack, extract_tensors
from vidlu.modules.tensor_extra import LogAbsDetJac, Name


def register_self_removing(register_proc, hook):
    handle_container = []

    @functools.wraps(hook)
    def wrapper(*args):
        result = hook(*args)
        handle_container[0].remove()
        return result

    handle_container.append(register_proc(wrapper))
    return handle_container[0]


# Hooks  ###########################################################################################

def check_no_inf_nan(module, input, output):
    """Invokes the debugger if it finds a NaN or inf in the output.

    Example:
        >>> for m in module.modules:
        >>>      m.register_forward_hook(check_no_inf_nan)
    """
    for output_index, out in enumerate(extract_tensors(output)):
        if (inf := torch.isinf(out).any()) or (nan := torch.isnan(out).any()):
            module_name = try_get_module_name_from_call_stack(module)
            print(f"{module_name=}, {output_index=} {inf=}, {nan=}")
            breakpoint()


def print_norm(module, input, output):
    """
    Example:
        >>> for m in module.modules():
        >>>      m.register_forward_hook(check_no_inf_nan)
    """
    module_name = try_get_module_name_from_call_stack(module)
    for output_index, out in enumerate(extract_tensors(input)):
        print(
            f"{module_name=}, [{output_index}] l2_norm={out.norm().item()}, linf_norm={out.abs().max()}")


def check_outputs_log_abs_det_jac(module, input, output):
    for i, y in enumerate(extract_tensors(output)):
        if not LogAbsDetJac.has(y):
            module_name = try_get_module_name_from_call_stack(module)
            raise RuntimeError(f"Output ({i}) of {module_name} ({type(module)}) has no"
                               + f" `ln(abs(det(jacobian)))` it.")
        else:
            print(Name.get(output, "noname"), try_get_module_name_from_call_stack(module))


def check_propagates_log_abs_det_jac(module, input, output):
    if LogAbsDetJac.has(next(extract_tensors(input))):
        check_outputs_log_abs_det_jac(module, input, output)
