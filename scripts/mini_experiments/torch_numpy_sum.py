import time

import torch

arrays = [torch.ones(1000, 1000).to("cuda:1") for _ in range(100)]
start = time.clock()
sum(arrays)
print(time.clock() - start)


def pasum(x):
    def sum_pairs(l, r):
        return [a + b for a, b in zip(l, r)]

    def split(x):
        l, r = x[:len(x) // 2], x[len(x) // 2:]
        r, rem = r[:len(l)], r[len(l):]
        return x, r, rem

    while len(x) > 1:
        l, r, rem = split(x)
        x = sum_pairs(l, r) + rem

    return x[0]


def timeit(n, proc):
    start = time.clock()
    for _ in range(n):
        proc()

    return time.clock() - start


print(timeit(100, lambda: pasum(arrays).shape))

s = None


def summ(newsum):
    global s
    s = newsum if s is None else s + newsum
    return s


import numpy as np

narrays = [np.ones((1000, 1000)) for _ in range(200)]
print(timeit(10, lambda: np.sum(np.stack(narrays), 0)))
print(timeit(10, lambda: np.sum(narrays, 0)))

# Torch ########################################################################

import time
import torch


def timeit(n, proc):
    start = time.clock()
    for _ in range(n):
        proc()
    torch.cuda.synchronize()
    return time.clock() - start


def py_sum(arrays):
    sum(arrays)


def stack_sum(arrays):
    torch.sum(torch.stack(arrays), 0)


n_repeats = 100

lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
py_sums = []
stack_sums = []

for n in lens:
    arrays = [torch.ones(8192 // n, 4096).to('cuda:0') for _ in range(n)]
    print(f'n = {n}:')
    py_sums += [timeit(n_repeats, lambda: py_sum(arrays))]
    print('     py-sum:', py_sums[-1])
    stack_sums += [timeit(n_repeats, lambda: stack_sum(arrays))]
    print('  stack-sum:', stack_sums[-1])

import matplotlib.pyplot as plt

plt.figure()
plt.xscale('log')

plt.plot(lens, py_sums, 'r', label='py-sum')
plt.plot(lens, stack_sums, 'g', label='stack-sum')
plt.legend()
plt.show()

# Numpy ########################################################################
import time
import numpy as np


def timeit(n, proc):
    start = time.clock()
    for _ in range(n):
        proc()
    return time.clock() - start


def np_sum(arrays):
    assert np.sum(arrays, 0).shape == arrays[0].shape


def np_stack_sum(arrays):
    assert np.sum(np.stack(arrays), 0).shape == arrays[0].shape


n_repeats = 10

lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
py_sums = []
np_sums = []
stack_sums = []

for n in lens:
    arrays = [np.ones((4096 // n, 4096)) for _ in range(n)]
    print(f'n = {n}:')
    py_sums += [timeit(n_repeats, lambda: py_sum(arrays))]
    print('     py-sum:', py_sums[-1])
    np_sums += [timeit(n_repeats, lambda: np_sum(arrays))]
    print('     np-sum:', np_sums[-1])
    stack_sums += [timeit(n_repeats, lambda: np_stack_sum(arrays))]
    print('  stack-sum:', stack_sums[-1])

import matplotlib.pyplot as plt

plt.figure()
plt.xscale('log')

plt.plot(lens, py_sums, 'r', label='py-sum')
plt.plot(lens, np_sums, 'b', label='np-sum')
plt.plot(lens, stack_sums, 'g', label='stack-sum')
plt.legend()
plt.show()

## Tracemalloc #################################################################

import time, torch
import torch.cuda
import tracemalloc


def b2mb(x): return int(x / 2 ** 20)


"""
class TorchTracemalloc():

    def start(self):
        self.begin = torch.cuda.memory_allocated(1)
        # torch.cuda.reset_peak_memory_stats()  # reset the peak gauge to zero
        return self

    def get_traced_memory(self, *exc):
        self.end = torch.cuda.memory_allocated(1)
        self.peak = torch.cuda.max_memory_allocated(1)
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        return self.used, self.peaked

    def stop(self):
        pass


tracemalloc = TorchTracemalloc()
"""


def timeit(n, proc):
    start = time.clock()
    for _ in range(n):
        _ = proc()
    torch.cuda.synchronize()
    return time.clock() - start


def py_sum_inplace(arrays):
    a = arrays[0]
    for i in range(1, len(arrays)):
        a += arrays[i]
    return a


def py_sum_inplace3(arrays):
    a = arrays[0]
    for i, x in enumerate(arrays):
        if i > 0:
            a += x
    return a


n_repeats = 100

lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
py_sums = []
py_sum_inplaces = []
stack_sums = []

mem = True

with torch.no_grad():
    for n in lens:
        arrays = [torch.ones(8192 // n, 8192) for _ in range(n)]
        # timeit(n_repeats, lambda: py_sum(arrays))

        print(f'n = {n}:')

        tracemalloc.start()
        py_sum_inplaces += [timeit(n_repeats, lambda: py_sum_inplace(arrays))]
        current, peak = tracemalloc.get_traced_memory()
        if mem:
            py_sum_inplaces[-1] = [peak]
        print(f"{current:0.2f}, {peak:0.2f}")
        tracemalloc.stop()
        print('  py-sum_in:', py_sum_inplaces[-1])
        torch.cuda.empty_cache()

        tracemalloc.start()
        py_sums += [timeit(n_repeats, lambda: py_sum(arrays))]
        current, peak = tracemalloc.get_traced_memory()
        if mem:
            py_sums[-1] = [peak]
        print(f"{current:0.2f}, {peak:0.2f}")
        tracemalloc.stop()
        print('     py-sum:', py_sums[-1])
        torch.cuda.empty_cache()

        tracemalloc.start()
        stack_sums += [timeit(n_repeats, lambda: stack_sum(arrays))]
        current, peak = tracemalloc.get_traced_memory()
        if mem:
            stack_sums[-1] = [peak]
        print(f"{current:0.2f}, {peak:0.2f}")
        tracemalloc.stop()
        print('  stack-sum:', stack_sums[-1])
        torch.cuda.empty_cache()

        for x in arrays:
            del x
        del arrays

import matplotlib.pyplot as plt

plt.figure()
plt.xscale('log')

plt.plot(lens, py_sums, 'r', label='builtins.sum')
plt.plot(lens, py_sum_inplaces, 'y', label='+=')
plt.plot(lens, stack_sums, 'g', label='stack,sum')
plt.legend()
plt.show()
