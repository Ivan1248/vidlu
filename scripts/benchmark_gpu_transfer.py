import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.cuda
from tqdm import tqdm, trange

import _context  # vidlu, dirs

from vidlu.utils.misc import Stopwatch
from vidlu.gpu_utils import get_available_gpu_indices

SHAPE = (512, 512, 128)
NUM_TRANSFERS = 400
NUM_MEASUREMENTS = 5
MULTIPROCESS = True

data_amount = NUM_TRANSFERS * np.prod(SHAPE).item() * torch.randn((1,)).element_size()

devices = [f'cuda:{i}' for i in range(len(get_available_gpu_indices()))]


# def run_single():
#     data_cpu = torch.randn(SHAPE)
#     data_gpu = {d: torch.randn(SHAPE, device=d) for d in devices}
#
#     sw = Stopwatch()
#
#     def transfer_to_gpu(dev, non_blocking):
#         return data_cpu.to(dev, non_blocking=non_blocking)
#
#     def transfer_from_gpu(dev, non_blocking):
#         return data_gpu[dev].to('cpu', non_blocking=non_blocking)
#
#     def benchmark(devices, proc):
#         measurements = []
#
#         for i in range(NUM_MEASUREMENTS):
#             torch.cuda.synchronize()
#             with sw.reset():
#                 for i in range(NUM_TRANSFERS):
#                     for d in devices:
#                         proc(d, non_blocking=i < NUM_TRANSFERS - 1)
#                         torch.cuda.empty_cache()
#                 torch.cuda.synchronize()
#
#             measurements.append(sw.time)
#
#         print(devices, proc.__name__, measurements)
#
#     for d in devices:
#         benchmark([d], transfer_to_gpu)
#     for d in devices:
#         benchmark([d], transfer_from_gpu)
#     benchmark(devices, transfer_to_gpu)
#     benchmark(devices, transfer_from_gpu)


def run_multi(dev, sem_ready: mp.Semaphore, sem_start: mp.Semaphore):
    data_cpu = torch.randn(SHAPE)
    data_gpu = torch.randn(SHAPE, device=dev)

    sw = Stopwatch()

    def transfer_to_gpu(non_blocking):
        return data_cpu.to(dev, non_blocking=non_blocking)

    def transfer_from_gpu(non_blocking):
        return data_gpu.to('cpu', non_blocking=non_blocking)

    def benchmark(proc):
        total_times = []

        for i in range(NUM_MEASUREMENTS):
            sem_ready.release()
            sem_start.acquire()
            # print('start', dev)
            torch.cuda.synchronize()
            with sw.reset():  # TODO: For some reason, a process can come here >2s after the first one.
                for i in range(NUM_TRANSFERS):
                    # print(f'[{sw.time:0.4f}s] {dev}: transfer {i}')
                    # TODO: Why is transfer from GPU faster when non_blocking=False?
                    proc(non_blocking=False)  # i < NUM_TRANSFERS - 1)
                torch.cuda.synchronize()

            total_times.append(sw.time)
        total_times = np.array(total_times)
        bandwidths = (data_amount / 1024 ** 3) / total_times
        avg_freqs = NUM_TRANSFERS / total_times

        print(f'{dev}: {proc.__name__}):',
              ', '.join(f'{b:0.2f}' for b in bandwidths),
              f'(median: {np.median(bandwidths):0.3f} GiB/s);',
              ', '.join(f'{b:0.2f}' for b in avg_freqs),
              f'(median: {np.median(avg_freqs):0.2f}/s)')

    benchmark(transfer_to_gpu)
    benchmark(transfer_from_gpu)


def synchronize_multi(n, sem_ready: mp.Semaphore, sem_start: mp.Semaphore):
    for _ in range(2):
        for _ in trange(NUM_MEASUREMENTS):
            for _ in range(n):
                sem_ready.acquire()
            for _ in range(n):
                sem_start.release()


if MULTIPROCESS:
    sem_ready = mp.Semaphore()
    sem_finished = mp.Semaphore()

    processes = {dev: mp.Process(target=run_multi, args=(f'{dev}', sem_ready, sem_finished))
                 for dev in devices}
    sync_process = mp.Process(target=synchronize_multi,
                              args=(len(processes), sem_ready, sem_finished))

    sync_process.start()
    for dev, process in processes.items():
        process.start()
    for dev, process in processes.items():
        process.join()
    sync_process.join()
