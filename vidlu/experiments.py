from argparse import Namespace
from functools import partial
from dataclasses import dataclass
import warnings

import torch
import numpy as np

from vidlu import gpu_utils, factories, defaults
from vidlu.modules import parameter_count
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.func import Empty
from vidlu.utils.indent_print import indent_print
from vidlu.utils.logger import Logger
from vidlu.utils.path import to_valid_path
from vidlu.utils import tree
from vidlu.utils.misc import try_input, query_yes_no


# TODO: logger instead of verbosity

@dataclass
class TrainingExperimentFactoryArgs:
    data: str
    input_prep: str
    model: str
    trainer: str
    metrics: str
    experiment_suffix: str
    resume: bool
    device: torch.device
    verbosity: int


# Data #############################################################################################

def get_prepared_data_for_trainer(data_str: str, datasets_dir, cache_dir, input_prep_str):
    data = factories.get_data(data_str, datasets_dir, cache_dir)
    data_flat = tree.flatten(data)
    if len(data_flat) != 2:
        raise ValueError(f'There must be exactly 2 datasets in "{data_str}"')
    ds_train, ds_test = dict(data_flat).values()
    ds_train = ds_train.map(defaults.get_jitter(ds_train))
    ds_train_jittered = ds_train.map(defaults.get_jitter(ds_train))

    prepare_input = factories.get_input_preparation(input_prep_str, ds_train)
    if not callable(prepare_input):
        raise RuntimeError("Not supported.")
    # if len(prepare_input) not in [1, len(data)]:
    #    raise ValueError("The number of input_preparation transforms should be"
    #                     + " either 1 or the number of different datasets (not subsets).")
    # data_flat = [((name, sub), ds.map(prepare_input[name])) for name, subsets in data.items() for
    #             sub, ds in subsets]

    # ds_train, ds_train_jittered, ds_test = (ds.map(prepare_input, func_name=input_prep_str)
    #                                        for ds in [ds_train, ds_train_jittered, ds_test])
    ds_train, ds_train_jittered, ds_test = map(prepare_input,
                                               [ds_train, ds_train_jittered, ds_test])

    return Namespace(train=ds_train, train_jittered=ds_train_jittered, test=ds_test)


# Model ############################################################################################

def get_model(model_str: str, ds_train, device, verbosity):
    model = factories.get_model(model_str, dataset=ds_train, verbosity=verbosity)
    model.to(device=device)
    if verbosity > 1:
        print(model)
    print('Parameter count:', parameter_count(model))
    return model


# Training loop actions ############################################################################

def define_training_loop_actions(trainer, cpman, ds_test, logger):
    @trainer.training.epoch_started.add_handler
    def on_epoch_started(es):
        logger.log(f"Starting epoch {es.epoch}/{es.max_epochs}"
                   + f" ({es.batch_count} batches, lr={trainer.lr_scheduler.get_lr()})")

    @trainer.training.epoch_completed.add_handler
    def on_epoch_completed(_):
        trainer.eval(ds_test)
        cpman.save(trainer.state_dict(), summary=logger.state_dict())  # checkpoint

    def report_metrics(es, is_validation=False):
        def eval_str(metrics):
            return ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])

        metrics = trainer.get_metric_values(reset=True)
        with indent_print():
            epoch_fmt, iter_fmt = f'{len(str(es.max_epochs))}d', f'{len(str(es.batch_count))}d'
            iter = es.iteration % es.batch_count
            prefix = ('val' if is_validation
                      else f'{format(es.epoch, epoch_fmt)}.'
                           + f'{format((iter - 1) % es.batch_count + 1, iter_fmt)}')
            logger.log(f"{prefix}: {eval_str(metrics)}")

    @trainer.training.iteration_completed.add_handler
    def on_iteration_completed(es):
        if es.iteration % es.batch_count % (es.batch_count // 5) == 0:
            report_metrics(es)

        optional_input = try_input()
        if optional_input is not None:
            try:
                exec(try_input())
            except:
                pass

    trainer.evaluation.epoch_completed.add_handler(partial(report_metrics, is_validation=True))


# Checkpoint manager ###############################################################################

def get_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_dir):
    a = training_args
    learner_name = to_valid_path(f"{a.model}/{a.trainer}")
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{a.data}/{learner_name}/{expsuff}'
    print('Learner name:', learner_name)
    print('Experiment ID:', experiment_id)
    return CheckpointManager(checkpoints_dir, experiment_str=experiment_id,
                             experiment_desc=training_args, resume=a.resume,
                             remove_old=a.experiment_suffix == '' and not a.resume)


# Experiment #######################################################################################
import torch.nn as nn
from vidlu.training import Trainer


@dataclass
class TrainingExperiment:
    model: nn.Module
    trainer: Trainer
    data: Namespace
    logger: Logger
    cpman: CheckpointManager

    @staticmethod
    def from_args(training_args: TrainingExperimentFactoryArgs, dirs):
        a = training_args
        with indent_print("Selecting device..."):
            if a.device is None:
                device = torch.device(
                        gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))
        with indent_print('Initializing data...'):
            data = get_prepared_data_for_trainer(a.data, dirs.DATASETS, dirs.CACHE, a.input_prep)
        with indent_print('Initializing model...'):
            model = get_model(a.model, data.train, device, a.verbosity)
        with indent_print('Initializing trainer and evaluation...'):
            trainer = factories.get_trainer(a.trainer, model=model, dataset=data.train,
                                            verbosity=a.verbosity)
            for m in factories.get_metrics(a.metrics, trainer, dataset=data.train):
                trainer.add_metric(m())
        logger = Logger()
        cpman = get_checkpoint_manager(a, dirs.SAVED_STATES)
        define_training_loop_actions(trainer, cpman, data.test, logger)

        if a.resume:
            for obj, state in zip([trainer, logger], cpman.load_last()):
                obj.load_state_dict(state)
            logger.print_all()
            model.to(device=device)

        return TrainingExperiment(model, trainer, data, logger, cpman)
