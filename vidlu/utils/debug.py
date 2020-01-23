import warnings
import traceback
import sys
from datetime import datetime

from vidlu.utils.indent_print import indent_print


def warning_debug_exec(code: str):
    warnings.warn("Debugging with ")
    with indent_print():
        print(code)
    print('in')
    with indent_print():
        traceback.print_stack()
    return exec(code)


# trace_calls

def trace_calls():
    def tracefunc(frame, event, arg, indent=(0,)):
        if event == "call":
            indent[0] += 2
            print("-" * indent[0] + "> call function", frame.f_code.co_name)
        elif event == "return":
            print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
            indent[0] -= 2
        return tracefunc

    sys.settrace(tracefunc)


def crash_after(proc, *datetime_args):
    if datetime.utcnow() <= datetime(*datetime_args):
        proc()
    raise AssertionError("")
