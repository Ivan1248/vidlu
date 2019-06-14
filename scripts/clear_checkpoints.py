import argparse

# noinspection PyUnusedImport
import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"

from _context import vidlu
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.path import to_valid_path

import dirs

# Arguments ########################################################################################

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('data', type=str)
parser.add_argument('model', type=str)
parser.add_argument('trainer', type=str)
parser.add_argument("-e", "--experiment_suffix", type=str, default='',
                    help="Experiment ID suffix. Required for running multiple experiments with the"
                         + " same configuration.")

args = parser.parse_args()

# Checkpoint manager - removing checkpoints ########################################################

learner_name = to_valid_path(f"{args.model}-{args.trainer}")
experiment_id = f'{args.data}-{learner_name}-exp_{args.experiment_suffix}'
print('Learner name:', learner_name)
print('Experiment ID:', experiment_id)
CheckpointManager(dirs.SAVED_STATES, experiment_str=experiment_id).remove_old_checkpoints(0)
