import subprocess
import warnings
from argparse import Namespace
import os
import time

import numpy as np


def nvidia_smi():
    import xmltodict
    xml_output = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE) \
        .communicate()[0]
    return xmltodict.parse(xml_output)['nvidia_smi_log']


def set_cuda_visible_devices(arg):
    os.environ['CUDA_VISIBLE_DEVICES'] = arg


def get_gpu_statuses(measurement_count=1, measuring_interval=1.0):
    if "CUDA_DEVICE_ORDER" not in os.environ or os.environ["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
        raise RuntimeError('The environment variable "CUDA_DEVICE_ORDER" should correspond to '
                           'nvidia-smi\'s device order, "PCI_BUS_ID".')

    def get_processes(gpu):
        if gpu['processes'] is None:
            return []
        processes = gpu['processes']['process_info']
        if not isinstance(processes, list):
            processes = [processes]
        return [Namespace(**p) for p in processes]

    if measurement_count > 1:
        measuring_period = measuring_interval / measurement_count
        samples = [get_gpu_statuses()]
        for _ in range(measurement_count - 1):
            time.sleep(measuring_period)
            samples.append(get_gpu_statuses())
        mem_total = samples[0].mem_total
        mem_used = max(s.mem_used for s in samples[0].mem_total)
        return Namespace(name=samples[-1].name,
                         gpu_util=sum(s.gpu_util for s in samples) / len(samples),
                         mem_total=mem_total, mem_used=mem_used, mem_free=mem_total - mem_used,
                         processes=samples[-1].processes)

    gpus = nvidia_smi()['gpu']
    if not isinstance(nvidia_smi()['gpu'], list):  # in case there is only 1 GPU
        gpus = [gpus]

    return [Namespace(name=gpu['product_name'],
                      gpu_util=float(gpu['utilization']['gpu_util'].split()[0]) / 100,
                      mem_total=int(gpu['fb_memory_usage']['total'].split()[0]),
                      mem_used=int(gpu['fb_memory_usage']['used'].split()[0]),
                      mem_free=int(gpu['fb_memory_usage']['free'].split()[0]),
                      processes=get_processes(gpu))
            for gpu in gpus]


def get_first_available_cuda_gpu(max_gpu_util, min_mem_free, no_processes=True):
    statuses = get_gpu_statuses()

    def availability_score(mem_free, gpu_util, process_count):
        return np.tan(np.arctan(mem_free * (1 - gpu_util)) - np.pi / 2 * int(process_count > 0))

    availabilities = [availability_score(s.mem_free, s.gpu_util, len(s.processes))
                      for s in statuses]
    if len(availabilities) == 0:
        raise RuntimeError("No cuda GPU-s found.")
    best_idx = int(np.argmax(availabilities))
    if ((statuses[best_idx].gpu_util > max_gpu_util)
            or (statuses[best_idx].mem_free < min_mem_free)
            or (no_processes and availabilities[best_idx] < 0)):
        best_idx = None
    return best_idx, statuses, availabilities


def get_first_available_device(max_gpu_util=0.3, min_mem_free=4000, no_processes=True, verbosity=0):
    try:
        device_idx, statuses, availabilities = get_first_available_cuda_gpu(
            max_gpu_util, min_mem_free, no_processes=no_processes)
        if verbosity > 0:
            print(f"Selected device {statuses[device_idx].name}"
                  + f" with availability score {availabilities[device_idx]}.")
        return device_idx
    except RuntimeError as e:
        warnings.warn(f"Unable to get GPU(s). \nCaught exception: \n{e}")
        if verbosity > 0:
            print(f"Selected device: CPU.")
        return 'cpu'
