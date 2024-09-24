# Based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py.
# Assumes that torchrun is used for multi-node training.

import os

import torch
import torch.distributed as dist

"""
The get_*_rank and get_*_world_size functions assume that torchrun is used to create processes and 
the environment.
https://pytorch.org/docs/stable/elastic/run.html
"""

IS_MULTI_NODE = 'GROUP_RANK' in os.environ


def distributed_is_enabled() -> bool:
    """Returns whether distributed training is initialized.

    Distributed training is initialized if init_process_group has been called."""
    return dist.is_available() and dist.is_initialized()


def get_group_rank(multi_node=IS_MULTI_NODE):
    """Returns the rank of the worker group. A number between 0 and max_nnodes. When running a
    single worker group per node, this is the rank of the node."""
    if not multi_node or not distributed_is_enabled():
        return 0
    return int(os.environ['GROUP_RANK'])


def get_global_size(multi_node=IS_MULTI_NODE) -> int:
    """Returns the number of all processes on all nodes."""
    if not distributed_is_enabled():
        return 1
    if multi_node:
        return int(os.environ['WORLD_SIZE'])
    return dist.get_world_size()


def get_local_size(multi_node=IS_MULTI_NODE) -> int:
    """Returns the size of the local (per-machine) process group."""
    if not distributed_is_enabled():
        return 1
    if multi_node:
        return int(os.environ['LOCAL_WORLD_SIZE'])
    return dist.get_world_size()


def get_global_rank(multi_node=IS_MULTI_NODE) -> int:
    """Returns the rank of the current process within all processes on all nodes."""
    if not distributed_is_enabled():
        return 0
    if multi_node:
        return int(os.environ['RANK'])
    return dist.get_rank()


def get_local_rank(multi_node=IS_MULTI_NODE) -> int:
    """Returns the rank of the current process within the local (per-machine) process group.

    This should be usually used only for assigning a device within the node."""
    if not distributed_is_enabled():
        return 0
    if multi_node:
        return int(os.environ['LOCAL_RANK'])
    return dist.get_rank()


def is_main_process() -> bool:
    return get_global_rank() == 0


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training

    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
    """
    if not distributed_is_enabled():
        return
    if get_global_size() == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings. It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
