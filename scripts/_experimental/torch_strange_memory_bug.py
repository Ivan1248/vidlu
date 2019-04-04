import torch
import tracemalloc


def idexing(arrays):
    for i in range(len(arrays)):
        arrays[i]


def list_copy(arrays):
    for x in arrays[:]:  # <- BUG HERE
        pass


def get_peak_mem_usage(proc):
    tracemalloc.start()
    proc()
    current, peak = tracemalloc.get_traced_memory()
    print(f"{current:0.2f}, {peak:0.2f}")
    tracemalloc.stop()
    torch.cuda.empty_cache()
    return peak


lens = [2 ** i for i in range(16)]
name_to_procedure = dict(idexing=idexing, list_copy=list_copy)
name_to_mem_usages = {k: [] for k in name_to_procedure.keys()}

with torch.no_grad():
    for n in lens:
        arrays = [torch.ones(2**16 // n, 4096).to('cuda:0') for _ in range(n)]

        print(f'n = {n}:')
        for name, proc in name_to_procedure.items():
            name_to_mem_usages[name] += [get_peak_mem_usage(lambda: proc(arrays))]

import matplotlib.pyplot as plt

plt.figure()
plt.xscale('log')

for name, mem_usages in name_to_mem_usages.items():
    plt.plot(lens, mem_usages, label=name)

plt.legend()
plt.show()
