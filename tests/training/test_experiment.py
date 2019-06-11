from pathlib import Path

import pytest

from argparse import Namespace

from vidlu.experiments import TrainingExperiment


def get_dirs(tmpdir):
    tmpdir = Path(tmpdir)
    dirs = dict(DATASETS=tmpdir / 'datasets', CACHE=tmpdir / 'cache',
                SAVED_STATES=tmpdir / 'states', PRETRAINED=tmpdir / 'pretrained_parameters')
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    return Namespace(**dirs)


def test_training_experiment(tmpdir):
    e = TrainingExperiment.from_args(
        data_str="DummyClassification{train,val}",
        input_prep_str="nop",
        model_str="DenseNet,backbone_f=t(depth=121,small_input=True)",
        trainer_str="Trainer,**{**configs.densenet_cifar,**dict(epoch_count=2)}",
        metrics_str="", experiment_suffix="", resume="", device=None, verbosity=1,
        dirs=get_dirs(tmpdir))

    e.trainer.eval(e.data.test)
    e.trainer.train(e.data.train_jittered, restart=False)

    e.cpman.remove_old_checkpoints()
