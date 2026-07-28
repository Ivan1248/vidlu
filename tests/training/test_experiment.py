from pathlib import Path

import pytest
from argparse import Namespace

from vidlu.experiments import (TrainingExperiment, TrainingExperimentFactoryArgs,
                               training_progress_step)
from vidlu.training.trainers import IterState


def get_dirs(tmpdir):
    tmpdir = Path(tmpdir)
    dirs = dict(datasets=tmpdir / 'datasets', cache=tmpdir / 'cache',
                saved_states=tmpdir / 'states', pretrained=tmpdir / 'pretrained')
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    return Namespace(**dirs)


def test_training_progress_step():
    assert training_progress_step(IterState()) == 0  # before training, batch_count unknown
    assert training_progress_step(IterState(batch_count=10, epoch=2, iteration=3)) == 23


@pytest.mark.skip(reason="Fatal Python error: Aborted in GitHub Actions.")
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
            resume_best=False,
            device=None,
            verbosity=1,
            restart=True),
        dirs=get_dirs(tmpdir))

    e.trainer.eval(e.data.test)
    e.trainer.train(e.data.train, restart=False)

    e.cpman.remove_old_checkpoints()
