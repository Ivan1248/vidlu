import subprocess
import xmltodict
from argparse import Namespace
import os

import numpy as np


def nvidia_smi():
    xml_output = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE) \
        .communicate()[0]
    return xmltodict.parse(xml_output)['nvidia_smi_log']


def get_gpu_statuses():
    if "CUDA_DEVICE_ORDER" not in os.environ or os.environ["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
        raise RuntimeError("CUDA device order does not correspond to nvidia-smi's device order.")

    def get_processes(gpu):
        if gpu['processes'] is None:
            return []
        processes = gpu['processes']['process_info']
        if not isinstance(processes, list):
            processes = [processes]
        return [Namespace(**p) for p in processes]

    return [Namespace(name=gpu['product_name'],
                      gpu_util=float(gpu['utilization']['gpu_util'].split()[0]) / 100,
                      mem_total=int(gpu['fb_memory_usage']['total'].split()[0]),
                      mem_used=int(gpu['fb_memory_usage']['used'].split()[0]),
                      mem_free=int(gpu['fb_memory_usage']['free'].split()[0]),
                      processes=get_processes(gpu))
            for gpu in nvidia_smi()['gpu']]


def get_first_available_cuda_gpu(max_gpu_util, min_mem_free, no_processes=True):
    statuses = get_gpu_statuses()

    def availability_score(mem_free, gpu_util, process_count):
        return np.tan(np.arctan(mem_free * (1 - gpu_util)) - np.pi / 2 * int(process_count > 0))

    availabilities = [availability_score(s.mem_free, s.gpu_util, len(s.processes))
                      for s in statuses]
    best_idx = int(np.argmax(availabilities)) if len(availabilities) > 0 else None
    if ((statuses[best_idx].gpu_util > max_gpu_util)
            or (statuses[best_idx].mem_free < min_mem_free)
            or (no_processes and availabilities[best_idx] < 0)):
        best_idx = None
    return best_idx, statuses, availabilities


def get_first_available_device(max_gpu_util=0.1, min_mem_free=4000):
    device_idx, statuses, availabilities = get_first_available_cuda_gpu(max_gpu_util, min_mem_free)
    if device_idx is None:
        print(f"Selected device: CPU.")
        return 'cpu'
    else:
        print(f"Selected device {statuses[device_idx].name}"
              + f" with availability score {availabilities[device_idx]}.")
        return device_idx
