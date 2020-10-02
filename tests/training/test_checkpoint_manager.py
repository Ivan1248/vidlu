import pytest

from functools import partial

from vidlu.training.checkpoint_manager import CheckpointManager


@pytest.mark.parametrize("scores,kept", [({1: 2, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}, {2, 3, 5, 6})])
def test_training_experiment(tmpdir, scores, kept):
    cpman_f = partial(CheckpointManager, checkpoints_dir=tmpdir, experiment_name="test",
                      info=dict(name="Foo"), n_last_kept=2, n_best_kept=2,
                      perf_func=lambda s: s['s'])
    cpman = cpman_f()
    for i, s in scores.items():
        cpman.save(dict(s=s), dict(i=i, s=s))

    with pytest.raises(RuntimeError):
        cpman2 = cpman_f()
    cpman2: CheckpointManager = cpman_f(resume=True)
    assert set(cpman2.saved) == set(map(str, kept))

    with pytest.raises(RuntimeError):
        cpman2.save(dict(s=10), dict(i=10, s=10))
    state, summary = cpman2.load_last()
    cpman2.save(dict(s=10), dict(i=10, s=10))

    cpman2.remove_checkpoints()

    with pytest.raises(RuntimeError):
        cpman3: CheckpointManager = cpman_f(resume=True)

    cpman3: CheckpointManager = cpman_f(resume=False)
    assert len(cpman3.saved) == 0
