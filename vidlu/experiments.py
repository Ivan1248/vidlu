from argparse import Namespace
from functools import partial
from dataclasses import dataclass
from pathlib import Path

import torch
from vidlu.data import Dataset
from IPython import embed

from vidlu import gpu_utils, factories, defaults, parameter_loading
from vidlu.data import Record
from vidlu.modules import parameter_count
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.indent_print import indent_print
from vidlu.utils.logger import Logger
from vidlu.utils.path import to_valid_path
from vidlu.utils import tree
from vidlu.utils.misc import try_input


# TODO: logger instead of verbosity

@dataclass
class TrainingExperimentFactoryArgs:
    data: str
    input_adapter: str
    model: str
    trainer: str
    metrics: str
    params: str
    experiment_suffix: str
    resume: bool
    device: torch.device
    verbosity: int


# Data #############################################################################################

def get_prepared_data_for_trainer(data_str: str, datasets_dir, cache_dir):
    data = factories.get_data(data_str, datasets_dir, cache_dir)
    datasets = tree.flatten(data)
    if len(datasets) != 2:
        raise ValueError(f'There must be exactly 2 datasets in "{data_str}"')
    ds_train, ds_test = dict(datasets).values()

    prepare_input = factories.get_input_preparation(ds_train)
    if not callable(prepare_input):
        raise RuntimeError("Not supported.")

    ds_train, ds_test = map(prepare_input, [ds_train, ds_test])

    return Namespace(train=ds_train, test=ds_test)


# Model ############################################################################################

def get_model(model_str: str, input_adapter_str, ds_train, device, verbosity):
    model = factories.get_model(model_str, input_adapter_str=input_adapter_str, dataset=ds_train,
                                verbosity=verbosity)

    model.to(device=device)
    if verbosity > 1:
        print(model)
    print('Parameter count:', parameter_count(model))
    return model


# Parameters #######################################################################################

def load_parameters(model, params_str, params_dir):
    model_name, params_name = params_str.split(',')
    state_dict = parameter_loading.get_parameters(model_name, Path(params_dir) / params_name)
    model.load_state_dict(state_dict)


# Training loop actions ############################################################################

def define_training_loop_actions(trainer, cpman, data, logger):
    @trainer.training.epoch_started.add_handler
    def on_epoch_started(es):
        logger.log(f"Starting epoch {es.epoch}/{es.max_epochs}"
                   + f" ({es.batch_count} batches, lr={trainer.lr_scheduler.get_lr()})")

    @trainer.training.epoch_completed.add_handler
    def on_epoch_completed(_):
        trainer.eval(data.test)
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

    @trainer.evaluation.iteration_completed.add_handler
    def on_eval_iteration_completed(state):
        nonlocal trainer, data

        optional_input = try_input()
        if optional_input is not None:
            try:
                exec(optional_input)
            except:
                print(f'Cannot execute "{optional_input}".')
                pass

    @trainer.training.iteration_completed.add_handler
    def on_iteration_completed(es):
        if es.iteration % es.batch_count % (max(1, es.batch_count // 5)) == 0:
            report_metrics(es)

        on_eval_iteration_completed(es)

    trainer.evaluation.epoch_completed.add_handler(partial(report_metrics, is_validation=True))


# Checkpoint manager ###############################################################################

def get_checkpoint_manager(training_args: TrainingExperimentFactoryArgs, checkpoints_dir):
    a = training_args
    learner_name = to_valid_path(f"{a.model}/{a.trainer}")
    expsuff = a.experiment_suffix or "_"
    experiment_id = f'{a.data}/{learner_name}/{expsuff}'
    print('Learner name:', learner_name)
    print('Experiment ID:', experiment_id)
    return CheckpointManager(checkpoints_dir, experiment_name=experiment_id,
                             experiment_desc=training_args, resume=a.resume,
                             remove_old=expsuff == '_' and not a.resume)


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
                a.device = torch.device(
                    gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))
        with indent_print('Initializing data...'):
            data = get_prepared_data_for_trainer(a.data, dirs.DATASETS, dirs.CACHE)
        with indent_print('Initializing model...'):
            model = get_model(a.model, a.input_adapter, data.train, a.device, a.verbosity)
        with indent_print('Initializing trainer and evaluation...'):
            trainer = factories.get_trainer(a.trainer, model=model, dataset=data.train,
                                            verbosity=a.verbosity)
            for m in factories.get_metrics(a.metrics, trainer, dataset=data.train):
                trainer.add_metric(m())
        logger = Logger()
        cpman = get_checkpoint_manager(a, dirs.SAVED_STATES)
        define_training_loop_actions(trainer, cpman, data, logger)

        if a.resume:
            for obj, state in zip([trainer, logger], cpman.load_last()):
                obj.load_state_dict(state)
            logger.print_all()
            model.to(device=a.device)

        if a.params is not None:
            load_parameters(model, params_str=a.params, params_dir=dirs.PRETRAINED)

        return TrainingExperiment(model, trainer, data, logger, cpman)


def ikmo_get_cityscapes_val(datasets_dir, cache_dir):
    from pathlib import Path
    from torchvision.transforms import Compose

    from vidlu.libs.swiftnet.data import Cityscapes as IKCityscapes

    from vidlu.libs.swiftnet.data.transform import Open, RemapLabels, Normalize, Tensor
    from vidlu.libs.swiftnet.data.mux.transform import Pyramid, SetTargetSize

    data_path = Path("/home/igrubisic/data/datasets/Cityscapes")

    # mydata = Cityscapes(data_path, subset="val")
    ts = (2048, 1024)
    ikdata = IKCityscapes(
        data_path,
        subset="val",
        transforms=Compose([Open(),
                            RemapLabels(IKCityscapes.map_to_id, IKCityscapes.num_classes),
                            Pyramid(alphas=[1.]),
                            SetTargetSize(target_size=ts,
                                          target_size_feats=(ts[0] // 4, ts[1] // 4)),
                            Normalize(scale=255, mean=IKCityscapes.mean, std=IKCityscapes.std),
                            Tensor(),
                            ]))

    def remap_labels(y):
        y[y == 19] = -1
        return y

    return Dataset(data=ikdata).map(
        lambda x: Record(x=x['image'], y=remap_labels(x['labels'])))
