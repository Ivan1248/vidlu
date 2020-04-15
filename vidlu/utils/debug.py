import warnings
import traceback
import sys
from datetime import datetime

from vidlu.utils.indent_print import indent_print


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


def crash_after(*args, format='%Y-%m-%d', message=None):
    if len(args) == 1 and isinstance(args[0], str):
        crashtime = datetime.strptime(args[0], format)
    elif len(args) > 1 and isinstance(args[0], int):
        if isinstance(args[-1], str):
            message, args = args[-1], args[:-1]
        crashtime = datetime(*args)
    else:
        crashtime = args[0]
    message = "" if message is None else f" Message: {message}"
    if datetime.utcnow() > crashtime:
        raise AssertionError(
            f"crash_after is crashing because the time {crashtime} has passed. {message}")
    else:
        warnings.warn(
            f"crash_after is not crashing because the time {crashtime} has not passed. {message}")
