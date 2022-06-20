import os
import warnings
from argparse import Namespace
from vidlu.utils.func import partial
from dataclasses import dataclass
from pathlib import Path
import typing as T
import numpy as np
import time

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from vidlu import factories
import vidlu.modules as vm
from vidlu.training import Trainer, CheckpointManager
from vidlu.utils.misc import indent_print
from vidlu.utils.logger import Logger
from vidlu.utils.path import to_valid_path
from vidlu.utils.misc import try_input, Stopwatch, query_user


# TODO: logger instead of verbosity

@dataclass
class TrainingExperimentFactoryArgs:
    data: str
    input_adapter: str
    model: str
    trainer: str
    metrics: str
    params: T.Optional[str]
    experiment_suffix: str
    resume: T.Optional[T.Literal["strict", "?", "best", "restart"]]
    device: T.Optional[torch.device]
    verbosity: int
    deterministic: bool


# Component factories (or factory wrappers) ########################################################


def get_report_iters(eval_count, iter_count, type_=set):
    """Evenly distributes 0-based iteration indices."""
    if eval_count > iter_count:
        return type_(range(eval_count))
    return type_(np.unique(np.linspace(0.5, iter_count + 0.5, eval_count + 1, dtype=int)[1:] - 1))


def to_dhm_str(time):
    time = int(time)
    d, time_d = divmod(time, 24 * 60 * 60)
    h, time_h = divmod(time_d, 60 * 60)
    m = time_h // 60
    return " ".join([f"{v}{k}" for k, v in dict(d=d, h=h, m=m).items() if v > 0])


# def find_best_epoch(epoch_to_main_metrics, metrics_to_perf):
#     epochs = list(epoch_to_main_metrics.keys())
#     perfs = list(map(metrics_to_perf, epoch_to_main_metrics.values()))
#     argmax = np.argmax(perfs)
#     return epochs[argmax]


def define_training_loop_actions(
        trainer: Trainer, cpman: CheckpointManager, data, logger, main_metrics: T.Sequence[str],
        eval_count=200, min_train_report_count=800,
        interact_shortcuts=dict(i='embed()', skip='loop.terminate()'),
        special_format={'mem': lambda v: f'{v}MiB', 'freq': lambda v: f'{v:.1f}/s'},
        line_width=120):
    sleepiness = 0
    eval_epochs = get_report_iters(eval_count, trainer.epoch_count)
    epoch_time, inter_epoch_time, eval_time = -1, -1, -1
    epoch_sw, inter_epoch_sw, eval_sw = Stopwatch(), Stopwatch(), Stopwatch()

    # epoch_to_main_metrics = cpman.load_last()

    def report_metrics(es, is_validation=False, line_width=line_width,
                       special_format: T.Mapping[str, T.Callable[[str], str]] = None,
                       prefix=None):
        special_format = special_format or {}

        def eval_str(metrics):
            def fmt(v):
                with np.printoptions(precision=2, threshold=20 if is_validation else 4,
                                     linewidth=line_width, floatmode='maxprec_equal',
                                     suppress=True):
                    return (f"{v:.4f}".lstrip('0') if isinstance(v, float) and v > 1e-3 else
                            f"{v:.2e}" if isinstance(v, float) else
                            f"\n{v}" if isinstance(v, np.ndarray) and v.ndim > 1 else
                            str(v).replace('0.', '.'))

            parts, line_len = [], 0
            for k, v in metrics.items():
                parts.append(special_format[k](v) if k in special_format else f"{k}={fmt(v)}")
                if len(lines := parts[-1].splitlines()) > 1 or \
                        line_len + len(parts[-1]) > line_width:
                    line_len = len(lines[-1])
                    parts[-1] = f"\n{parts[-1]}"
            return ', '.join(parts)

        metrics = trainer.get_metric_values(reset=True)

        with indent_print():
            epoch_fmt, iter_fmt = f'{len(str(trainer.epoch_count))}d', f'{len(str(es.batch_count))}d'
            epoch = trainer.training.state.epoch
            iter_ = es.iteration % es.batch_count
            if prefix is None:
                prefix = (f'{format(epoch + 1, epoch_fmt)} val' if is_validation
                          else f'{format(epoch + 1, epoch_fmt)}.'
                               + f'{format(iter_ % es.batch_count + 1, iter_fmt)}')
            logger.log(f"{prefix}: {eval_str(metrics)}")
            # logger.log(f"Epoch to performance: {cpman.id_to_perf}")

    @trainer.training.epoch_started.handler
    def on_epoch_started(es):
        epoch_sw.start()
        if sleepiness > 0:
            print(f"Warning: {sleepiness}s of sleep per epoch.")
        time_left_training = (1 - es.epoch / es.max_epochs) * (es.max_epochs * epoch_time)
        time_left = time_left_training + (1 - es.epoch / es.max_epochs) * (eval_count * eval_time)
        info_str = (f"Epoch {es.epoch + 1}/{es.max_epochs}:"
                    + f" {es.batch_count} batches,"
                    + f" lr=({', '.join(f'{x:.2e}' for x in trainer.lr_scheduler.get_last_lr())}),"
                    + f" devices={{{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}}}")
        if epoch_time > 0:
            info_str += f", left {to_dhm_str(time_left)} ({to_dhm_str(time_left_training)} training)"
        logger.log(info_str)

    @trainer.training.epoch_completed.handler
    def on_epoch_completed(es):
        nonlocal epoch_time
        epoch_time = epoch_sw.time
        inter_epoch_sw.start()
        if es.epoch in eval_epochs:
            es_val = trainer.eval(data.test)
            # epoch_to_main_metrics[es.epoch] = {k: es_val.metrics[k] for k in main_metrics}
            # best_epoch = find_best_epoch(epoch_to_main_metrics, lambda s: s[main_metrics[0]])
            # report_metrics(es_val, special_format=special_format, is_validation=True,
            #                prefix=f'Best epoch ({best_epoch}): ')
            cpman.save(trainer.state_dict(),
                       summary=dict(logger=logger.state_dict(),
                                    perf=es_val.metrics[main_metrics[0]],
                                    # summary=epoch_to_main_metrics,
                                    log="\n".join(logger.lines),
                                    epoch=es.epoch))

    @trainer.training.iter_completed.handler
    def on_iteration_completed(es):
        report_iters = get_report_iters(max(1, min_train_report_count // trainer.epoch_count),
                                        es.batch_count)
        iter = es.iteration % es.batch_count
        if iter in report_iters:
            report_metrics(es, special_format=special_format)

        interact(es, loop=trainer.training)

        if sleepiness > 0:
            time.sleep(sleepiness / es.batch_count)

    @trainer.evaluation.iter_completed.handler
    def on_eval_iteration_completed(es):
        interact(es, loop=trainer.evaluation)

        if sleepiness > 0:
            time.sleep(sleepiness / es.batch_count)

    @trainer.evaluation.epoch_started.handler
    def on_eval_epoch_started(es):
        nonlocal inter_epoch_time
        inter_epoch_time = inter_epoch_sw.time
        eval_sw.start()

    @trainer.evaluation.epoch_completed.handler
    def on_eval_epoch_completed(es):
        nonlocal eval_time
        eval_time = eval_sw.time
        report_metrics(es, special_format=special_format, is_validation=True)

    def set_sleepiness(x):
        nonlocal sleepiness
        sleepiness = x

    # noinspection PyUnresolvedReferences
    def interact(es, loop):
        if (optional_input := try_input()) is None:
            return
        from IPython import embed
        import vidlu.utils.presentation.visualization as visualization
        nonlocal trainer, data, set_sleepiness, sleepiness, cpman, logger, main_metrics
        try:
            state = es
            cmd = interact_shortcuts.get(optional_input, optional_input)
            print(f"Iteration: {es.iteration}, namespace: " + ", ".join(locals().keys()))
            exec(cmd)
        except Exception as e:
            print(f'Cannot execute "{optional_input}". Error:\n{e}.')


# Experiment #######################################################################################

def get_device(device_str):
    if device_str is None:
        if torch.cuda.device_count() == 0 \
                and not query_user("No GPU found. Are you sure you want to continue?",
                                   timeout=10, default='y'):
            raise RuntimeError("No GPU available.")
        return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
    elif device_str == "auto":
        from vidlu import gpu_utils
        return torch.device(
            gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))


def get_experiment_name(training_args):
    a = training_args
    learner_name = to_valid_path(f"{a.input_adapter}/{a.model}/{a.trainer}"
                                 + (f"/{a.params}" if a.params else ""), split_long_names=True)
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{a.data}/{learner_name}/{expsuff}'
    return experiment_id


def get_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_dir):
    a = training_args
    experiment_id = get_experiment_name(training_args)
    cpman = CheckpointManager(
        checkpoints_dir, experiment_name=experiment_id, info=training_args,
        separately_saved_state_parts=("model",), n_best_kept=1,
        mode=('start' if a.resume is None else
              'restart' if a.resume == "restart" else
              'resume_or_start' if a.resume == "?" else
              'resume'),
        perf_func=lambda s: s.get('perf', 0), log_func=lambda s: s.get('log', ""),
        name_suffix_func=lambda s: f"{s['epoch']}_{s['perf']:.3f}")
    return cpman


def load_parameters(model, params_str, params_dir):
    parameters, dest = factories.get_translated_parameters(params_str=params_str,
                                                           params_dir=params_dir)
    module = vm.get_submodule(model, dest)
    try:
        module.load_state_dict(parameters, strict=True)
    except RuntimeError as e:
        warnings.warn(str(e))
        module.load_state_dict(parameters, strict=False)


def _check_dirs(dirs):
    for name in ['datasets', 'cache', 'saved_states', 'pretrained']:
        dirs_ = getattr(dirs, name)
        if isinstance(dirs_, (str, Path)):
            dirs_ = [dirs_]
        for d in dirs_:
            if d is None or not Path(d).is_dir():
                raise NotADirectoryError(f"dirs.{name}={getattr(dirs, name)} is not"
                                         f" a directory path or a sequence thereof.")


@dataclass
class TrainingExperiment:
    model: nn.Module
    trainer: Trainer
    data: Namespace
    cpman: CheckpointManager
    logger: Logger

    @staticmethod
    def from_args(training_args: TrainingExperimentFactoryArgs, dirs):
        _check_dirs(dirs)
        a = training_args
        logger = Logger()

        with indent_print("\nSetting device..."):
            a.device = get_device(a.device)
            print(f"device: {a.device}")

        with indent_print('\nInitializing checkpoint manager...'):
            cpman = get_checkpoint_manager(a, dirs.saved_states)

        try:
            with indent_print('\nInitializing data...'):
                print(a.data)
                with Stopwatch() as t:
                    data = factories.get_prepared_data_for_trainer(a.data, dirs.datasets,
                                                                   dirs.cache)
                print(f"Data initialized in {t.time:.2f} s.")
            first_ds = next(iter(data.values()))

            with indent_print('\nInitializing model...'):
                print(a.model)
                with Stopwatch() as t:
                    model = factories.get_model(a.model, input_adapter_str=a.input_adapter,
                                                prep_dataset=first_ds, device=a.device,
                                                verbosity=a.verbosity)
                print(f"Model initialized in {t.time:.2f} s.")

            with indent_print('\nInitializing trainer and evaluation...'):
                print(a.trainer)
                trainer = factories.get_trainer(a.trainer, model=model, dataset=first_ds,
                                                verbosity=a.verbosity,
                                                deterministic=a.deterministic)
                metrics, main_metrics = factories.get_metrics(a.metrics, trainer, dataset=first_ds)
                for m in metrics:
                    trainer.metrics.append(m())

            define_training_loop_actions(trainer, cpman, data, logger, main_metrics=main_metrics)
        except Exception:
            raise
        finally:
            resuming_required = cpman.resuming_required
            if resuming_required:
                state, summary = (cpman.load_best if a.resume == "best" else cpman.load_last)(
                    map_location=a.device)
                # TODO: remove backward compatibility
                logger.load_state_dict(summary.get('logger', summary))
                logger.print_all()

        if resuming_required:
            trainer.load_state_dict(state)
        elif a.params is not None:
            with indent_print("\nLoading parameters..."):
                print(a.params)
                load_parameters(model, a.params, dirs.pretrained)

        return TrainingExperiment(model, trainer, data, cpman, logger)
