import argparse

# noinspection PyUnresolvedReferences
import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"
import torch
import numpy as np

# noinspection PyUnresolvedReferences
from _context import vidlu
from vidlu import factories
from vidlu.experiments import TrainingExperiment, TrainingExperimentFactoryArgs
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.func import Empty, call_with_args_from_dict
from vidlu.utils.indent_print import indent_print

import dirs


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    e = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)

    if not args.no_init_test:
        print('Evaluating initially...')
        e.trainer.eval(e.data.test)

    print(('Continuing' if args.resume else 'Starting') + ' training...')
    training_datasets = {k: v for k, v in e.data.items() if k.startswith("train")}
    e.trainer.train(*training_datasets.values(), restart=False)
    print(f'Evaluating on training data ({", ".join(training_datasets.keys())})...')
    for name, ds in training_datasets.items():
        e.trainer.eval(ds)

    e.cpman.remove_old_checkpoints()

    print(f'Trained model saved in {e.cpman.experiment_dir}')


def test(args):
    te = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)

    print('Starting evaluation (test/val):...')
    te.trainer.eval(te.data.test)
    print('Starting evaluation (train):...')
    te.trainer.eval(te.data.train)

    te.cpman.remove_old_checkpoints()


def test_trained(args):
    e = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs,
                                 {**args.__dict__, **dict(resume=True, redo=False)}), dirs=dirs)

    print('Starting evaluation (test/val):...')
    e.trainer.eval(e.data.test)
    print('Starting evaluation (train):...')
    e.trainer.eval(e.data.train)

    e.cpman.remove_old_checkpoints()


# Argument parsing #################################################################################

def add_standard_arguments(parser, func):
    # learner configuration
    parser.add_argument("data", type=str, help=factories.get_data.help)
    parser.add_argument("input_adapter", type=str, default=Empty,
                        help='A string representing input adaptation to the model, '
                             + 'e.g. "standardize", "div255".')
    parser.add_argument("model", type=str, help=factories.get_model.help)
    parser.add_argument("trainer", type=str, help=factories.get_trainer.help)
    parser.add_argument("--params", type=str, default=None,
                        help='The name of the file containing parameters.')
    parser.add_argument("--metrics", type=str, default="",
                        help='A comma-separated list of metrics.')
    # device
    parser.add_argument("-d", "--device", type=torch.device, help="PyTorch device.", default=None)
    # experiment result saving, state checkpoints
    parser.add_argument("-e", "--experiment_suffix", type=str, default=None,
                        help="Experiment ID suffix. Required for running multiple experiments"
                             + " with the same configuration.")
    if func is train:
        parser.add_argument("-r", "--resume", action='store_true',
                            help="Resume training from the last checkpoint of the same experiment.")
        parser.add_argument("--redo", action='store_true',
                            help="Delete the data of an experiment with the same name.")
    parser.add_argument("--no_init_test", action='store_true',
                        help="Skip testing before training.")
    parser.add_argument("-s", "--seed", type=str, default=53,
                        help="RNG seed for experiment reproducibility. Useless on GPU.")
    # reporting
    parser.add_argument("-v", "--verbosity", type=int, help="Console output verbosity.",
                        default=1)
    parser.set_defaults(func=func)


parser = argparse.ArgumentParser(description='Experiment running script')
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser("train")
add_standard_arguments(parser_train, train)

parser_test = subparsers.add_parser("test")
add_standard_arguments(parser_test, test)

parser_test_trained = subparsers.add_parser("test_trained")
add_standard_arguments(parser_test_trained, test_trained)

args = parser.parse_args()

with indent_print("Arguments:"):
    print(args)

args.func(args)
