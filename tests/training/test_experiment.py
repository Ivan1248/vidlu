from pathlib import Path

import pytest

from argparse import Namespace

from vidlu.experiments import TrainingExperiment, TrainingExperimentFactoryArgs
import vidlu.training.configs as tc


def get_dirs(tmpdir):
    tmpdir = Path(tmpdir)
    dirs = dict(DATASETS=tmpdir / 'datasets', CACHE=tmpdir / 'cache',
                SAVED_STATES=tmpdir / 'states', PRETRAINED=tmpdir / 'pretrained_parameters')
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    return Namespace(**dirs)


def test_training_experiment(tmpdir):
    e = TrainingExperiment.from_args(
        TrainingExperimentFactoryArgs(
            data="DummyClassification(size=25){train,val}",
            input_adapter="id",
            model="DenseNet,backbone_f=t(depth=40,k=12,small_input=True)",
            trainer="tc.densenet_cifar,epoch_count=2,batch_size=4",
            metrics="",
            params=None,
            experiment_suffix="_",
            resume=False,
            device=None,
            verbosity=1,
            redo=True),
        dirs=get_dirs(tmpdir))

    e.trainer.eval(e.data.test)
    e.trainer.train(e.data.train, restart=False)

    e.cpman.remove_old_checkpoints()
