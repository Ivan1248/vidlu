import argparse

# noinspection PyUnresolvedReferences
import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"
import torch
import numpy as np

# noinspection PyUnresolvedReferences
from _context import vidlu
from vidlu import factories
from vidlu.experiments import TrainingExperiment
from vidlu.utils.func import Empty
from vidlu.utils.indent_print import indent_print

import dirs


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    e = TrainingExperiment.from_args(
        data_str=args.data, input_prep_str=args.input_prep, model_str=args.model,
        trainer_str=args.trainer, metrics_str=args.metrics,
        experiment_suffix=args.experiment_suffix, resume=args.resume, device=args.device,
        verbosity=args.verbosity, dirs=dirs)

    print(('Continuing' if args.resume else 'Starting') + ' training:...')

    #e.trainer.eval(e.data.test)
    e.trainer.train(e.data.train_jittered, restart=False)

    e.cpman.remove_old_checkpoints()

    print(f'Trained model saved in {e.cpman.dir_path}')


# Argument parsing #################################################################################

parser = argparse.ArgumentParser(description='Training script')
subparsers = parser.add_subparsers()
parser_train = subparsers.add_parser("train")
# learner configuration
parser_train.add_argument("data", type=str, help=factories.get_data.help)
parser_train.add_argument("input_prep", type=str, default=Empty,
                          help='A string representing input preparation, e.g. "standardize", "div255".')
parser_train.add_argument("model", type=str, help=factories.get_model.help)
parser_train.add_argument("trainer", type=str, help=factories.get_trainer.help)
parser_train.add_argument("--metrics", type=str, default="",
                          help='A comma-separated list of metrics.')
# device
parser_train.add_argument("-d", "--device", type=torch.device, help="PyTorch device.", default=None)
# experiment result saving, state checkpoints
parser_train.add_argument("-e", "--experiment_suffix", type=str, default="",
                          help="Experiment ID suffix. Required for running multiple experiments"
                               + " with the same configuration.")
parser_train.add_argument("-r", "--resume", action='store_true',
                          help="Resume training from the last checkpoint of the same experiment.")
parser_train.add_argument("-s", "--seed", type=str, default=53,
                          help="RNG seed for experiment reproducibility. Useless on GPU.")
# reporting
parser_train.add_argument("-v", "--verbosity", type=int, help="Console output verbosity.",
                          default=1)
parser_train.set_defaults(func=train)

args = parser.parse_args()

with indent_print("Arguments:"):
    print(args)

args.func(args)
