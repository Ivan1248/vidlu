import argparse

from _context import vidlu
from vidlu.utils.func import find_empty_args
from vidlu.utils.func import tryable

# example: python find_empty_args.py vidlu.learning.models.ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)
args = parser.parse_args()

namespace_str, func_str = args.func.rsplit('.', 1)
try:
    exec(f"from {namespace_str} import {func_str}")
except ImportError:
    exec(f"import  {namespace_str}")
    print(f"{namespace_str} doesn't contain {func_str}. It only contains the following:")
    namespace = eval(namespace_str)
    for x in dir(namespace):
        obj = getattr(namespace, x)
        if not x.startswith('_') and callable(obj) and \
                tryable((lambda: obj.__module__ == namespace_str), True)():
            print(f"  {x}")
    exit()

func = eval(func_str)

empty_args = find_empty_args(func)
n = len(empty_args)
print(f"Found {n} empty argument{'' if n == 1 else 's'} in {func_str}{'.' if n == 0 else ':'}")
for ea in empty_args:
    print('/'.join(ea))
