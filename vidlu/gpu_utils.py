import subprocess
import xmltodict
from argparse import Namespace
import os

import numpy as np

from vidlu.utils.tree import convert_tree


def nvidia_smi():
    xml_output = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE) \
        .communicate()[0]
    return convert_tree(xmltodict.parse(xml_output)['nvidia_smi_log'], dict, recurse_sequences=True)


def get_device_statuses():
    if "CUDA_DEVICE_ORDER" not in os.environ or os.environ["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
        raise EnvironmentError(
            "CUDA device order does not correspond to nvidia-smi's device order.")

    return [Namespace(name=x['product_name'],
                      gpu_util=float(x['utilization']['gpu_util'].split()[0]) / 100,
                      mem_total=int(x['fb_memory_usage']['total'].split()[0]),
                      mem_used=int(x['fb_memory_usage']['used'].split()[0]),
                      mem_free=int(x['fb_memory_usage']['free'].split()[0]))
            for x in nvidia_smi()['gpu']]


def get_first_available_cuda_gpu(max_gpu_util=0.1, min_mem_free=4000):
    statuses = get_device_statuses()
    availabilities = [x.mem_free * (1 - x.gpu_util) for x in statuses]
    best_idx = int(np.argmax(availabilities)) if len(availabilities) > 0 else None
    if max_gpu_util is not None:
        if statuses[best_idx].gpu_util > max_gpu_util:
            best_idx = None
    if best_idx is not None and min_mem_free is not None:
        if statuses[best_idx].mem_free < min_mem_free:
            best_idx = None
    return best_idx, statuses, availabilities


def get_device(*args, **kwargs):
    device_idx, statuses, availabilities = get_first_available_cuda_gpu(*args, **kwargs)
    if device_idx is None:
        print(f"Selected device: CPU.")
        return 'cpu'
    else:
        print(f"Selected device {statuses[device_idx].name}"
              + f" with availability score {availabilities[device_idx]}.")
        return device_idx
