import argparse

from _context import vidlu
from functools import partial, partialmethod
from vidlu.utils.misc import *
from vidlu.utils.func import *

# example: python find_empty_args.py vidlu.learning.models.ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('namespace', type=str)
parser.add_argument('func', type=str)
args = parser.parse_args()

namespace_str, func_str = args.namespace, args.func

exec(f"from {namespace_str} import *")

try:
    func = eval(func_str)
    empty_args = find_empty_args(func)
    n = len(empty_args)
    print(f"Found {n} empty argument{'' if n == 1 else 's'} in {func_str}{'.' if n == 0 else ':'}")
    for ea in empty_args:
        print(' ', '/'.join(ea))
    print("As tree:")
    print(' ', paths_to_tree([(x, '?') for x in empty_args], ArgTree))
except (NameError, AttributeError):
    print(f"{namespace_str} contains the following:")
    namespace = eval(namespace_str)
    for x in dir(namespace):
        obj = getattr(namespace, x)
        if not x.startswith('_') and callable(obj) and \
                tryable((lambda: obj.__module__ == namespace_str), True)():
            print(f"  {x}")
    raise
