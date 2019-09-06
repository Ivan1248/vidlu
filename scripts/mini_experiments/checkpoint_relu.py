import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial

mem = [torch.cuda.max_memory_allocated()]
d = []

x = torch.zeros(10 ** 4, 10 ** 4, device='cuda')
mem += [torch.cuda.max_memory_allocated()]
d += [mem[-1] - mem[-2]]

funcs = {
    #'dummy': lambda: 2 * x,
    #'inplace-relu': lambda: F.relu(x, inplace=True),  # 4
    'checkpoint-relu': lambda: checkpoint(F.relu, x),  # 8
    #'relu': lambda: F.relu(x),  # 8
}

y = {}

for k, func in funcs.items():
    y[k] = torch.tensor([2.], requires_grad=True, device='cuda') * func()
    mem += [torch.cuda.max_memory_allocated()]
    d += [mem[-1] - mem[-2]]

#print(torch.all(y['checkpoint-relu'] == y['relu']))
print([m // 10 ** 8 for m in mem])
print(d[0] // 10 ** 8)
print('\n'.join([f"{name}: {delta // 10 ** 8}" for name, delta in zip(funcs.keys(), d[1:])]))
