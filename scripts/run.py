import sys
import argparse
from time import time
import random
from datetime import datetime, timedelta
import os
import warnings
import contextlib as ctx

# noinspection PyUnresolvedReferences
# import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"
import torch
import numpy as np

# noinspection PyUnresolvedReferences
import _context  # vidlu, dirs

from vidlu import factories
from vidlu.experiments import TrainingExperiment, TrainingExperimentFactoryArgs
from vidlu.utils.func import Empty, call_with_args_from_dict
from vidlu.utils.misc import indent_print
from vidlu.utils.misc import query_user
from vidlu.utils import debug
from vidlu.data import clean_up_dataset_cache
import dirs


def log_run(status):
    try:
        import fcntl
    except ImportError as ex:
        from unittest.mock import Mock
        fcntl = Mock()

    if (d := len('start') - len(status)) > 0:
        status += ' ' * d
    try:
        with (dirs.experiments / 'runs.txt').open('a') as runs_file:
            prefix = f"[{status} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            fcntl.flock(runs_file, fcntl.LOCK_EX)
            args = [a if i < 2 or len(a) == 0 or a[0] == '-' else f'"{a}"'
                    for i, a in enumerate(sys.argv)]
            runs_file.write(f"{prefix} {' '.join(args)}\n")
            fcntl.flock(runs_file, fcntl.LOCK_UN)
    except IOError as e:
        warnings.warn(str(e))
        traceback.print_exc()
        print(e)


def train(args):
    if args.debug:
        debug.trace_calls(depth=122,
                          filter_=lambda frame, *a, **k: "vidlu" in frame.f_code.co_filename
                                                         and not frame.f_code.co_name[0] in "_<")

    if args.restart and not query_user("Are you sure you want to restart the experiment?",
                                       timeout=30, default='y'):
        exit()

    seed = int(time()) % 100 if args.seed is None else args.seed  # 53
    for rseed in [torch.manual_seed, np.random.seed, random.seed]:
        rseed(seed)

    exp = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)

    exp.logger.log(f"RNG seed: {seed}")

    # if args.debug:
    #     debug.stop_tracing_calls()

    profiler = None
    if args.profile:
        from torch.autograd.profiler import profile
        from functools import partial
        profiler = partial(profile, use_cuda=True)

    with profiler() if profiler is not None else ctx.suppress() as prof:
        if not args.no_init_eval:
            print('\nEvaluating initially...')
            exp.trainer.eval(exp.data.test)
        log_run('cont.' if args.resume else 'start')

        print(('\nContinuing' if args.resume else 'Starting') + ' training...')
        training_datasets = {k: v for k, v in exp.data.items() if k.startswith("train")}

        exp.trainer.train(*training_datasets.values(), restart=False)

        if not args.no_train_eval:
            print(f'\nEvaluating on training data ({", ".join(training_datasets.keys())})...')
            for name, ds in training_datasets.items():
                exp.trainer.eval(ds)
        log_run('done')
    if prof is not None:
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    exp.cpman.remove_old_checkpoints()

    print(f"\nRNG seed: {seed}")
    print(f'State saved in\n{exp.cpman.last_checkpoint_path}')

    if dirs.cache is not None:
        cache_cleanup_time = int(os.environ.get("VIDLU_DATA_CACHE_CLEANUP_TIME", 60))
        clean_up_dataset_cache(dirs.cache / 'datasets', timedelta(days=cache_cleanup_time))


def path(args):
    e = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)
    print(e.cpman.experiment_dir)


def test(args):
    if args.restart:
        raise ValueError("`restart=True` is not allowed in the test procedure.")
    if not args.resume:
        warnings.warn("`resume` is set to `False`. The initial parameters will be tested.")
    e = TrainingExperiment.from_args(
        call_with_args_from_dict(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)

    if (module_arg := args.module) is not None:
        import importlib
        module_name, proc_name, *_ = *module_arg.split(':'), None
        if proc_name is None:
            proc_name = 'run'

        module = importlib.import_module(module_name)
        if ',' in proc_name:
            proc_name, args_str = proc_name.split(",", 1)
            result = eval(f"{proc_name}({args_str})", vars(module), locals())
        else:
            result = getattr(module, proc_name)(e)

    else:
        print('Starting evaluation (test/val):...')
        e.trainer.eval(e.data.test)
        print('Starting evaluation (train):...')
        e.trainer.eval(e.data.train)


# Argument parsing #################################################################################

def add_standard_arguments(parser, func):
    # learner configuration
    parser.add_argument("data", type=str, help=factories.get_data.help)
    parser.add_argument("input_adapter", type=str, default=Empty,
                        help='A string representing input adaptation to the model, '
                             + 'e.g. "id", "standardize".')
    parser.add_argument("model", type=str, help=factories.get_model.help)
    parser.add_argument("trainer", type=str, help=factories.get_trainer.help)
    parser.add_argument("--evaluator", type=str, help=factories.get_trainer.help)
    parser.add_argument("--params", type=str, default=None,
                        help='The name of the file containing parameters.')
    parser.add_argument("--metrics", type=str, default="",
                        help='A comma-separated list of metrics.')
    # device
    parser.add_argument("-d", "--device", type=str, help="PyTorch device.",
                        default=None)
    # experiment result saving, state checkpoints
    parser.add_argument("-e", "--experiment_suffix", type=str, default=None,
                        help="Experiment ID suffix. Required for running multiple experiments"
                             + " with the same configuration.")
    parser.add_argument("-rb", "--resume_best", action='store_true',
                        help="Use the best checkpoint if --resume is provided.")
    # if func is train:
    cpman_mode = parser.add_mutually_exclusive_group(required=False)
    cpman_mode.add_argument("-r", "--resume", action='store_true',
                            help="Resume training from a checkpoint of the same experiment.")
    cpman_mode.add_argument("-rs", "--resume_or_start", action='store_true',
                            help="Resume training if there is a checkpoint or start new training.")
    cpman_mode.add_argument("--restart", action='store_true',
                            help="Delete the data of an experiment with the same name.")
    parser.add_argument("--no_init_eval", action='store_true',
                        help="Skip testing before training.")
    parser.add_argument("--no_train_eval", action='store_true',
                        help="No evaluation on the training set.")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed. Default: int(time()) %% 100.")
    # reporting, debugging
    parser.add_argument("--debug", help="Enable autograd anomaly detection.", action='store_true')
    parser.add_argument("--profile", help="Enable CUDA profiling.", action='store_true')
    parser.add_argument("--warnings_as_errors", help="Raise errors instead of warnings.",
                        action='store_true')
    parser.add_argument("-v", "--verbosity", type=int, help="Console output verbosity.", default=2)
    parser.set_defaults(func=func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment running script')
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser("train")
    add_standard_arguments(parser_train, train)

    parser_train = subparsers.add_parser("path")
    add_standard_arguments(parser_train, path)

    parser_test = subparsers.add_parser("test")
    add_standard_arguments(parser_test, test)
    parser_test.add_argument("-m", "--module", type=str, default=None,
                             help="Path of a module containing a run(Experiment) procedure.")

    args = parser.parse_args()

    with indent_print("Arguments:"):
        print(args)

    if args.debug:
        print("Debug: Autograd anomaly detection on.")
        torch.autograd.set_detect_anomaly(True)

    if args.warnings_as_errors:
        import traceback


        def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
            log = file if hasattr(file, 'write') else sys.stderr
            traceback.print_stack(file=log)
            log.write(warnings.formatwarning(message, category, filename, lineno, line))


        warnings.showwarning = warn_with_traceback
        # warnings.simplefilter("always")

    args.func(args)
