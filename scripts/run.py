import sys
import argparse
import random
from datetime import datetime, timedelta
import os
import warnings
import contextlib as ctx
from pathlib import Path
import subprocess
import shlex
import time
import traceback

import torch
import numpy as np

# noinspection PyUnresolvedReferences
import _context  # vidlu, dirs

import dirs
from vidlu import factories
import vidlu.experiments as ve
from vidlu.experiments import TrainingExperiment, TrainingExperimentFactoryArgs
from vidlu.utils.func import Empty, call_with_assignable_args
from vidlu.utils.misc import indent_print, query_user
from vidlu.utils import debug
import vidlu.torch_utils as vtu
from vidlu.data import clean_up_dataset_cache


def log_run(status, result=None):
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
            if result is not None:
                runs_file.write(f"{result}\n")
            runs_file.write(f"\n")
            fcntl.flock(runs_file, fcntl.LOCK_UN)
    except IOError as e:
        warnings.warn(str(e))
        traceback.print_exc()
        print(e)


def fetch_remote_experiment(args, dirs):
    dir = shlex.quote(str(Path(dirs.saved_states) / ve.get_experiment_name(args)))
    cmd = ["rsync", "-azvO", "--relative", "--delete", f"{args.remote_experiments}:{dir}/",
           f"/"]
    print("Running " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        warnings.warn(f"Experiment state transfer had errors: {result}")
        query_user("Experiment state transfer had errors. Continue?", default='n')


def get_profiler():
    from torch.autograd.profiler import profile
    return profile(use_cuda=True)


def make_experiment(args, dirs):
    return TrainingExperiment.from_args(
        call_with_assignable_args(TrainingExperimentFactoryArgs, args.__dict__), dirs=dirs)


@torch.no_grad()
def eval_with_pop_stats(exp, stats_dataset):
    """Evaluates with an approximation of populations stats in batchnorms"""
    with vtu.preserve_state(exp.trainer.model):
        for m in exp.trainer.model.modules():
            if "Norm" in type(m).__name__ and hasattr(m, "track_running_stats"):
                m.momentum = None
                m.track_running_stats = True
                m.num_batches_tracked *= 0
        exp.trainer.model.train()
        print(f'\nComputing approximate population statistics...')
        exp.trainer.eval(stats_dataset)
        exp.trainer.model.eval()
        print(f'\nEvaluating using approximate population statistics...')
        exp.trainer.eval(exp.data.test)


def train(args):
    if args.resume == "restart" \
            and not query_user("Are you sure you want to restart the experiment?",
                               timeout=30, default='y'):
        exit()

    if args.remote_experiments and args.resume not in [None, "restart"]:
        fetch_remote_experiment(args, dirs)

    exp = make_experiment(args, dirs=dirs)

    exp.logger.log("Resume command:\n\x1b[0;30;42m"
                   + f'run.py train "{args.data}" "{args.input_adapter}" "{args.model}"'
                   + f' "{args.trainer}" --params "{args.params}" -d {repr(args.device)} --metrics "{args.metrics}"'
                   + f' -e {args.experiment_suffix or "_"} -r\x1b[0m')
    exp.logger.log(f"RNG seed: {args.seed}")

    with get_profiler() if args.profile else ctx.suppress() as prof:
        if not args.no_init_eval:
            print('\nEvaluating initially...')
            exp.trainer.eval(exp.data.test)
        log_run('cont.' if args.resume else 'start')

        print(('\nContinuing' if args.resume not in (
            "restart", None) else 'Starting') + ' training...')
        training_datasets = {k: v for k, v in exp.data.items() if k.startswith("train")}

        torch.cuda.empty_cache()
        exp.trainer.train(*training_datasets.values(), restart=False)

        if args.eval_with_pop_stats:
            eval_with_pop_stats(exp, exp.data.train)

        if not args.no_train_eval:
            print(f'\nEvaluating on training data ({", ".join(training_datasets.keys())})...')
            for name, ds in training_datasets.items():
                exp.trainer.eval(ds)

        print(exp.cpman.id_to_perf)
        log_run('done', str(exp.cpman.id_to_perf))

    if args.profile:
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    exp.cpman.remove_old_checkpoints()

    print(f"\nRNG seed: {args.seed}")
    print(f'State saved in\n{exp.cpman.last_checkpoint_path}')

    if dirs.cache is not None:
        cache_cleanup_time = int(os.environ.get("VIDLU_DATA_CACHE_CLEANUP_TIME", 60))
        clean_up_dataset_cache(dirs.cache / 'datasets', timedelta(days=cache_cleanup_time))


def get_path(args):
    e = make_experiment(args, dirs=dirs)
    print(e.cpman.experiment_dir)


def test(args):
    if not args.resume:
        warnings.warn("`resume` is set to `False`. The initial parameters will be tested.")

    e = make_experiment(args, dirs=dirs)

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
    parser.add_argument("--eval_with_pop_stats", action='store_true',
                        help="Computes actual population statistics for batchnorm layers.")
    # device
    parser.add_argument("-d", "--device", type=str, help="PyTorch device.",
                        default=None)
    # experiment result saving, state checkpoints
    parser.add_argument("-e", "--experiment_suffix", type=str, default=None,
                        help="Experiment ID suffix. Required for running multiple experiments"
                             + " with the same configuration.")
    parser.add_argument("--remote_experiments", type=str, default=None,
                        help="Identifier of experiment directory to copy the state from for resuming")
    parser.add_argument("-r", "--resume", action='store', nargs="?",
                        choices=["strict", "?", "best", "restart"], default=None, const="strict",
                        help="Resume training from checkpoint of the same experiment. "
                             + "? - can start new training if there are no checkpoints, "
                             + "best - resumes from the best checkpoint, "
                             + "restart - deletes existing checkpoints and restarts the experiments.")
    parser.add_argument("--no_init_eval", action='store_true',
                        help="Skip testing before training.")
    parser.add_argument("--no_train_eval", action='store_true',
                        help="No evaluation on the training set.")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="RNG seed. Default: int(time()) %% 100.")
    parser.add_argument("--deterministic", action='store_true',
                        help="Usage of deterministic operations.")
    # reporting, debugging
    parser.add_argument("--debug", help="", action='store_true')
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

    parser_get_path = subparsers.add_parser("get_path")
    add_standard_arguments(parser_get_path, get_path)

    parser_test = subparsers.add_parser("test")
    add_standard_arguments(parser_test, test)
    parser_test.add_argument("-m", "--module", type=str, default=None,
                             help="Path of a module containing a run(Experiment) procedure.")

    args = parser.parse_args()

    debug.set_traceback_format(call_pdb=args.debug, verbose=args.debug)

    with indent_print("Arguments:"):
        print(args)

    if args.deterministic:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    seed = (0 if args.deterministic else int(time.time()) % 100) if args.seed is None else args.seed
    for rseed in [torch.manual_seed, np.random.seed, random.seed]:
        rseed(seed)
    args.seed = seed

    if args.debug:
        print("Debug: Autograd anomaly detection on.")
        torch.autograd.set_detect_anomaly(True)
        debug.trace_calls(depth=122,
                          filter_=lambda frame, *a, **k: "vidlu" in frame.f_code.co_filename
                                                         and not frame.f_code.co_name[0] in "_<")

    if args.warnings_as_errors:
        debug.set_warnings_with_traceback()

    args.func(args)
