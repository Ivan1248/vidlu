import warnings
from argparse import Namespace
from functools import partial
from dataclasses import dataclass
from pathlib import Path
import typing as T
import numpy as np

import torch
import torch.nn as nn

from vidlu import factories
import vidlu.modules as vm
from vidlu.training import Trainer, CheckpointManager
from vidlu.utils.indent_print import indent_print
from vidlu.utils.logger import Logger
from vidlu.utils.path import to_valid_path
from vidlu.utils.misc import try_input, Stopwatch


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
    redo: bool
    device: T.Optional[torch.device]
    verbosity: int


# Component factories (or factory wrappers) ########################################################


def define_training_loop_actions(trainer: Trainer, cpman, data, logger):
    @trainer.training.epoch_started.handler
    def on_epoch_started(es):
        logger.log(f"Starting epoch {es.epoch}/{es.max_epochs}"
                   + f" ({es.batch_count} batches,"
                   + f" lr={', '.join(f'{x:.2e}' for x in trainer.lr_scheduler.get_lr())})")

    @trainer.training.epoch_completed.handler
    def on_epoch_completed(es):
        if es.epoch % max(1, len(data.test) // len(data.train)) == 0 \
                or es.epoch == es.max_epochs - 1:
            trainer.eval(data.test)
            cpman.save(trainer.state_dict(), summary=logger.state_dict())  # checkpoint

    def report_metrics(es, is_validation=False):
        def eval_str(metrics):
            def fmt(v):
                return (f"{v:.4f}".lstrip('0') if isinstance(v, float) else
                        (f"\n{v}" if v.ndim > 1 else v) if isinstance(v, np.ndarray) else
                        str(v))

            with np.printoptions(precision=2, threshold=20 if is_validation else 4, linewidth=120,
                                 floatmode='maxprec_equal', suppress=True):
                return ', '.join([f"{fmt(v)}MiB" if k == 'mem' else
                                  f"{fmt(v)}/s" if k == 'freq' else
                                  f"{k}={fmt(v)}" for k, v in metrics.items()])

        metrics = trainer.get_metric_values(reset=True)
        with indent_print():
            epoch_fmt, iter_fmt = f'{len(str(es.max_epochs))}d', f'{len(str(es.batch_count))}d'
            iter_ = es.iteration % es.batch_count
            prefix = ('val' if is_validation
                      else f'{format(es.epoch, epoch_fmt)}.'
                           + f'{format((iter_ - 1) % es.batch_count + 1, iter_fmt)}')
            logger.log(f"{prefix}: {eval_str(metrics)}")

    # noinspection PyUnresolvedReferences
    @trainer.evaluation.iteration_completed.handler
    def interact(state):
        from IPython import embed
        from vidlu.utils.presentation import visualization
        nonlocal trainer, data

        optional_input = try_input()
        if optional_input is not None:
            if optional_input == 'i':
                optional_input = 'embed()'
            try:
                exec(optional_input)
            except Exception as ex:
                print(f'Cannot execute "{optional_input}"\n{ex}.')

    @trainer.training.iteration_completed.handler
    def on_iteration_completed(es):
        if es.iteration % es.batch_count % (max(1, es.batch_count // 5)) == 0:
            remaining = es.batch_count - es.iteration % es.batch_count
            if remaining >= es.batch_count // 5 or remaining == 0:
                report_metrics(es)

        interact(es)

    trainer.evaluation.epoch_completed.add_handler(partial(report_metrics, is_validation=True))


def get_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_dir):
    a = training_args
    learner_name = to_valid_path(f"{a.model}/{a.trainer}"
                                 + (f"/{a.params}" if a.params else ""))
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{a.data}/{learner_name}/{expsuff}'
    print('Learner name:', learner_name)
    print('Experiment ID:', experiment_id)
    return CheckpointManager(checkpoints_dir, experiment_name=experiment_id,
                             experiment_desc=training_args, resume=a.resume,
                             remove_old=a.redo)


# Experiment #######################################################################################

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
        with indent_print("Selecting device..."):
            if a.device is None:
                a.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # a.device = torch.device(
                #    gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))
            print(f"device: {a.device}")
        with indent_print('Initializing data...'):
            with Stopwatch() as t:
                data = factories.get_prepared_data_for_trainer(a.data, dirs.DATASETS, dirs.CACHE)
            print(f"Data initialized in {t.time:.2f} s.")
        with indent_print('Initializing model...'):
            with Stopwatch() as t:
                model = factories.get_model(a.model, input_adapter_str=a.input_adapter,
                                            prep_dataset=next(iter(data.values())), device=a.device,
                                            verbosity=a.verbosity)
            print(f"Model initialized in {t.time:.2f} s.")
        with indent_print('Initializing trainer and evaluation...'):
            trainer = factories.get_trainer(a.trainer, model=model,
                                            dataset=next(iter(data.values())),
                                            verbosity=a.verbosity)
            for m in factories.get_metrics(a.metrics, trainer, dataset=next(iter(data.values()))):
                trainer.metrics.append(m())
        logger = Logger()
        logger.log("Resume command:\n"
                   + f'run.py train "{a.data}" "{a.input_adapter}" "{a.model}" "{a.trainer}"'
                   + f' -d "{a.device}" --metrics "{a.metrics}" -r')
        cpman = get_checkpoint_manager(a, dirs.SAVED_STATES)
        define_training_loop_actions(trainer, cpman, data, logger)

        if a.resume:
            for obj, state in zip([trainer, logger], cpman.load_last(map_location=a.device)):
                obj.load_state_dict(state)
            logger.print_all()
        elif a.params is not None:
            parameters, dest = factories.get_translated_parameters(params_str=a.params,
                                                                   params_dir=dirs.PRETRAINED)
            module = vm.get_submodule(model, dest)
            try:
                module.load_state_dict(parameters, strict=True)
            except RuntimeError as ex:
                warnings.warn(str(ex))
                module.load_state_dict(parameters, strict=False)
        return TrainingExperiment(model, trainer, data, logger, cpman)
