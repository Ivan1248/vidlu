import warnings
import traceback

from vidlu.utils.indent_print import indent_print


def warning_debug_exec(code: str):
    warnings.warn("Debugging with ")
    with indent_print():
        print(code)
    print('in')
    with indent_print():
        traceback.print_stack()
    return exec(code)
