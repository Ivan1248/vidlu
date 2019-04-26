import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

# noinspection PyUnusedImport
import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"
import torch

from _context import vidlu
from vidlu.modules.elements import parameter_count
from vidlu.factories import parse_datasets, parse_model, parse_trainer, parse_metrics
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.indent_print import indent_print
from vidlu.utils.path import to_valid_path
from vidlu import gpu_utils, defaults

import dirs

# Arguments ########################################################################################

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('data', type=str, help=parse_datasets.help)
parser.add_argument('model', type=str, help=parse_model.help)
parser.add_argument('trainer', type=str, help=parse_trainer.help)
parser.add_argument('evaluation', type=str, default="",
                    help='A comma-separated list of metrics.')
parser.add_argument("-r", "--resume", action='store_true',
                    help="Resume training from the last checkpoint of the same experiment.")
parser.add_argument("-e", "--experiment_suffix", type=str, default='',
                    help="Experiment ID suffix. Required for running multiple experiments with the"
                         + " same configuration.")

args = parser.parse_args()

# Checkpoint manager - removing checkpoints ########################################################

learner_name = to_valid_path(f"{args.model}-{args.trainer}")
experiment_id = f'{args.data}-{learner_name}-exp_{args.experiment_suffix}'
print('Learner name:', learner_name)
print('Experiment ID:', experiment_id)
CheckpointManager(dirs.SAVED_STATES, id=experiment_id).remove_old_checkpoints(0)
