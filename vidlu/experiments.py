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
    resume: bool
    resume_best: bool
    restart: bool
    device: T.Optional[torch.device]
    verbosity: int


# Component factories (or factory wrappers) ########################################################


def get_report_iters(eval_count, iter_count, type_=set):
    """Evenly distributes 0-based iteration indices."""
    if eval_count > iter_count:
        return type_(range(eval_count))
    return type_(np.unique(np.linspace(0.5, iter_count + 0.5, eval_count + 1, dtype=int)[1:] - 1))


def define_training_loop_actions(
        trainer: Trainer, cpman: CheckpointManager, data, logger, main_metrics: T.Sequence[str],
        eval_count=200, min_train_report_count=800, interact_shortcuts=dict(i='embed()'),
        special_format={'mem': lambda v: f'{v}MiB', 'freq': lambda v: f'{v:.1f}/s'},
        line_width=120):
    sleepiness = 0
    eval_epochs = get_report_iters(eval_count, trainer.epoch_count)

    @trainer.training.epoch_started.handler
    def on_epoch_started(es):
        if sleepiness > 0:
            print(f"Warning: {sleepiness}s of sleep per epoch.")
        logger.log(f"Epoch {es.epoch + 1}/{es.max_epochs}:"
                   + f" {es.batch_count} batches,"
                   + f" lr=({', '.join(f'{x:.2e}' for x in trainer.lr_scheduler.get_last_lr())})")

    @trainer.training.epoch_completed.handler
    def on_epoch_completed(es):
        if es.epoch in eval_epochs:
            es_val = trainer.eval(data.test)
            cpman.save(trainer.state_dict(),
                       summary=dict(logger=logger.state_dict(),
                                    perf=es_val.metrics[main_metrics[0]],
                                    log="\n".join(logger.lines),
                                    epoch=es.epoch))

    def report_metrics(es, is_validation=False, line_width=line_width,
                       special_format: T.Mapping[str, T.Callable[[str], str]] = None):
        special_format = special_format or {}

        def eval_str(metrics):
            def fmt(v):
                with np.printoptions(precision=2, threshold=20 if is_validation else 4,
                                     linewidth=line_width, floatmode='maxprec_equal',
                                     suppress=True):
                    return (f"{v:.4f}".lstrip('0') if isinstance(v, float) else
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
            prefix = (f'{format(epoch + 1, epoch_fmt)} val' if is_validation
                      else f'{format(epoch + 1, epoch_fmt)}.'
                           + f'{format(iter_ % es.batch_count + 1, iter_fmt)}')
            logger.log(f"{prefix}: {eval_str(metrics)}")
            # logger.log(f"Epoch to performance: {cpman.id_to_perf}")

    def set_sleepiness(x):
        nonlocal sleepiness
        sleepiness = x

    # noinspection PyUnresolvedReferences
    @trainer.evaluation.started.handler
    @trainer.evaluation.iter_completed.handler
    def interact(state):
        if (optional_input := try_input()) is None:
            return

        from IPython import embed
        from vidlu.utils.presentation import visualization
        nonlocal trainer, data, set_sleepiness, sleepiness
        try:
            cmd = interact_shortcuts.get(optional_input, optional_input)
            print(f"Variables: " + ", ".join(locals().keys()))
            exec(cmd)
        except Exception as e:
            print(f'Cannot execute "{optional_input}". Error:\n{e}.')

    @trainer.training.iter_completed.handler
    def on_iteration_completed(es):
        report_iters = get_report_iters(max(1, min_train_report_count // trainer.epoch_count),
                                        es.batch_count)
        iter = es.iteration % es.batch_count
        if iter in report_iters:
            report_metrics(es, special_format=special_format)

        interact(es)

        if sleepiness > 0:
            time.sleep(sleepiness / es.batch_count)

    trainer.evaluation.epoch_completed.add_handler(
        partial(report_metrics, special_format=special_format, is_validation=True))


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


def get_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_dir):
    a = training_args
    learner_name = to_valid_path(f"{a.model}/{a.trainer}"
                                 + (f"/{a.params}" if a.params else ""))
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{a.data}/{learner_name}/{expsuff}'
    print('Learner name:', learner_name)
    print('Experiment ID:', experiment_id)
    cpman = CheckpointManager(checkpoints_dir, experiment_name=experiment_id,
                              info=training_args, separately_saved_state_parts=("model",),
                              n_best_kept=1,
                              mode='restart' if a.restart else 'resume' if a.resume else 'new',
                              perf_func=lambda s: s.get('perf', 0),
                              log_func=lambda s: s.get('log', ""),
                              name_suffix_func=lambda s: f"{s['epoch']}_{s['perf']:.3f}")
    return cpman


def _check_dirs(dirs):
    for name in ['DATASETS', 'CACHE', 'SAVED_STATES', 'PRETRAINED']:
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
    logger: Logger
    cpman: CheckpointManager

    @staticmethod
    def from_args(training_args: TrainingExperimentFactoryArgs, dirs):
        _check_dirs(dirs)
        a = training_args

        with indent_print("Setting device..."):
            a.device = get_device(a.device)
            print(f"device: {a.device}")

        with indent_print('Initializing checkpoint manager and logger...'):
            logger = Logger()
            logger.log("Resume command:\n"
                       + f'run.py train "{a.data}" "{a.input_adapter}" "{a.model}" "{a.trainer}"'
                       + f' -d "{a.device}" --metrics "{a.metrics}" -r')

            cpman = get_checkpoint_manager(a, dirs.SAVED_STATES)

        try:
            with indent_print('Initializing data...'):
                print(a.data)
                with Stopwatch() as t:
                    data = factories.get_prepared_data_for_trainer(a.data, dirs.DATASETS,
                                                                   dirs.CACHE)
                print(f"Data initialized in {t.time:.2f} s.")
            first_ds = next(iter(data.values()))

            with indent_print('Initializing model...'):
                print(a.model)
                with Stopwatch() as t:
                    model = factories.get_model(a.model, input_adapter_str=a.input_adapter,
                                                prep_dataset=first_ds, device=a.device,
                                                verbosity=a.verbosity)
                print(f"Model initialized in {t.time:.2f} s.")

            with indent_print('Initializing trainer and evaluation...'):
                print(a.trainer)
                trainer = factories.get_trainer(a.trainer, model=model, dataset=first_ds,
                                                verbosity=a.verbosity)
                metrics, main_metrics = factories.get_metrics(a.metrics, trainer, dataset=first_ds)
                for m in metrics:
                    trainer.metrics.append(m())

            define_training_loop_actions(trainer, cpman, data, logger, main_metrics=main_metrics)
        except Exception as e:
            raise e
        finally:
            if a.resume:
                state, summary = (cpman.load_best if a.resume_best else cpman.load_last)(
                    map_location=a.device)
                # TODO: remove backward compatibility
                logger.load_state_dict(summary.get('logger', summary))
                logger.print_all()

        if a.resume:
            trainer.load_state_dict(state)
        elif a.params is not None:
            with indent_print("Loading parameters..."):
                print(a.params)
                parameters, dest = factories.get_translated_parameters(params_str=a.params,
                                                                       params_dir=dirs.PRETRAINED)
                module = vm.get_submodule(model, dest)
                try:
                    module.load_state_dict(parameters, strict=True)
                except RuntimeError as e:
                    warnings.warn(str(e))
                    module.load_state_dict(parameters, strict=False)
        return TrainingExperiment(model, trainer, data, logger, cpman)
