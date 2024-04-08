import warnings
import sys
import traceback
from datetime import datetime
import inspect
from argparse import Namespace
from IPython.core import ultratb

# STATE - any state should be kept here ############################################################
state = Namespace()


# ##################################################################################################


def trace_calls(depth=float('inf'),
                filter_=lambda **k: True,
                enter_action=lambda depth, frame, **k: print(
                    " " * (depth * 2), frame.f_code.co_name,
                    f"{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}")):
    max_depth, depth = depth, 0

    def trace_func(frame, event, arg):
        nonlocal depth
        # frame.f_trace_lines = False
        if depth < max_depth:
            if event == "call":
                depth += 1
                if filter_(frame=frame, event=event, arg=arg):
                    enter_action(frame=frame, event=event, arg=arg, depth=depth)
            elif event == "return":
                depth -= 1
            return trace_func
        return None

    sys.setprofile(trace_func)


def trace_lines(depth=float('inf'),
                filter_=lambda **k: True,
                enter_action=lambda depth, frame, **k: print(
                    " " * (depth * 2),
                    f"{frame.f_code.co_filename}:{frame.f_code.co_name}:{frame.f_lineno}")):
    max_depth, depth = depth, 0

    def trace_func(frame, event, arg):
        nonlocal depth
        # frame.f_trace_lines = True
        if depth < max_depth:
            if event == "call":
                depth += 1
            elif event == "return":
                depth -= 1
            elif event == 'line':
                if filter_(frame=frame, event=event, arg=arg):
                    enter_action(frame=frame, event=event, arg=arg, depth=depth)
            return trace_func
        return None

    sys.settrace(trace_func)


def stop_tracing_calls():
    sys.setprofile(None)
    sys.settrace(None)


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
            f"crash_after is crashing because the time {crashtime} has passed.{message}")
    else:
        warnings.warn(f"This code will crash after {crashtime}.{message}")


# Tracebacks

def set_traceback_format(call_pdb=False, verbose=False):
    sys.excepthook = ultratb.FormattedTB(mode='Verbose' if verbose else 'Plain',
                                         color_scheme='Linux', call_pdb=call_pdb)


def set_warnings_with_traceback():
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    # warnings.simplefilter("always")


# Keeping a state associated with a line in code

def here_code_line_info(caller_order=0):
    frame = inspect.stack()[caller_order + 1]
    return frame.filename, frame.lineno


def here_number_of_runs(id=None, increment=True):
    if id is None:
        id = here_code_line_info(1)
    num = state.get((here_number_of_runs, id), 0)
    if increment:
        state[(here_number_of_runs, id)] = num + 1
    return num


class _OldValue:
    pass


old_value = _OldValue


def here_state(id=None, new_value=old_value):
    if id is None:
        id = here_code_line_info(1)
    value = state.get((here_state, id), None)
    if new_value is not old_value:
        state[(here_state, id)] = new_value
    return value
