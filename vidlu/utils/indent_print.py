import contextlib
import builtins


@contextlib.contextmanager
def indent_print(*args, indent="   ", print_action=print):
    if len(args) > 0:
        print(*args)
    orig_print = builtins.print

    def ind_print(*args, **kwargs):
        print_action(indent[:-1], *args, **kwargs)

    builtins.print = ind_print
    yield
    builtins.print = orig_print
