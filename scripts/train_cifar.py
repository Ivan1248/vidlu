from functools import partial
import argparse

# noinspection PyUnresolvedReferences
import torch

# noinspection PyUnresolvedReferences
import _context  # vidlu, dirs

from vidlu import factories, models, modules
from vidlu.experiments import (TrainingExperiment, TrainingExperimentFactoryArgs,
                               define_training_loop_actions)
import vidlu.data as vd
from vidlu.utils.misc import indent_print
from vidlu.utils.logger import Logger
from vidlu.utils import debug
import vidlu.utils.func as vuf
import vidlu.configs.training as vct
from vidlu.training import CheckpointManager, Trainer
import dirs
from run import log_run


def get_experiment(resume=True, restart=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with indent_print('Initializing checkpoint manager and logger...'):
        logger = Logger()
        cpman = CheckpointManager(dirs.SAVED_STATES, experiment_name="resnet_cifar_example",
                                  separately_saved_state_parts=("model",), n_best_kept=1,
                                  mode='restart' if restart else 'resume' if resume else 'new',
                                  perf_func=lambda s: s.get('perf', 0),
                                  log_func=lambda s: s.get('log', ""),
                                  name_suffix_func=lambda s: f"{s['epoch']}_{s['perf']:.3f}")

    with indent_print('Initializing data...'):
        data = factories.get_prepared_data_for_trainer("Cifar10{trainval,test}", dirs.DATASETS,
                                                       dirs.CACHE)
    first_ds = next(iter(data.values()))

    with indent_print('Initializing model...'):
        t = vuf.ArgTree  # used to propagate arguments into nested functions
        model_f = vuf.argtree_partial(models.ResNetV1,
                                      backbone_f=t(depth=18, small_input=True),
                                      head_f=partial(modules.components.ClassificationHead,
                                                     class_count=first_ds.info.class_count))
        model = model_f()
        init_input = next(iter(vd.DataLoader(first_ds, batch_size=1)))[0]
        factories.build_and_init_model(model, init_input, device=device)

    with indent_print('Initializing trainer and evaluation...'):
        config = vct.TrainerConfig(vct.resnet_cifar,
                                   # epoch_count=100,
                                   # jitter=None,
                                   )
        trainer = factories.get_trainer("ct.resnet_cifar", dataset=first_ds, model=model)
        # trainer = Trainer(**config, loss=modules.losses.NLLLossWithLogits(ignore_index=-1),
        #                   model=model)
        metrics, main_metrics = factories.get_metrics("", trainer, dataset=first_ds)
        for m in metrics:
            trainer.metrics.append(m())

    define_training_loop_actions(trainer, cpman, data, logger, main_metrics=main_metrics)

    if resume:
        state, summary = cpman.load_last(map_location=device)
        logger.load_state_dict(summary.get('logger', summary))
        logger.print_all()
        trainer.load_state_dict(state)

    return TrainingExperiment(model, trainer, data, logger, cpman)


def train(resume, restart):
    exp = get_experiment(resume=resume, restart=restart)

    debug.stop_tracing_calls()

    print('Evaluating initially...')
    exp.trainer.eval(exp.data.test)
    log_run('cont.' if resume else 'start')

    print(('Continuing' if resume else 'Starting') + ' training...')
    training_datasets = {k: v for k, v in exp.data.items() if k.startswith("train")}
    exp.trainer.train(*training_datasets.values(), restart=False)

    print(f'Evaluating on training data ({", ".join(training_datasets.keys())})...')
    for name, ds in training_datasets.items():
        exp.trainer.eval(ds)
    log_run('done')

    exp.cpman.remove_old_checkpoints()

    print(f'State saved in\n{exp.cpman.last_checkpoint_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment running script')
    parser.add_argument("-r", "--resume", action='store_true',
                        help="Resume training from a checkpoint of the same experiment.")
    parser.add_argument("--restart", action='store_true',
                        help="Delete the data of an experiment with the same name.")
    subparsers = parser.add_subparsers()

    args = parser.parse_args()

    train(resume=args.resume, restart=args.restart)
