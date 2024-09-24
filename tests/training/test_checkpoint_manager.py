import pytest

from functools import partial

from vidlu.training.checkpoint_manager import CheckpointManager


@pytest.mark.parametrize("scores,kept", [({1: 2, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}, {2, 3, 5, 6})])
def test_training_experiment(tmpdir, scores, kept):
    cpman_f = partial(CheckpointManager, checkpoints_dir=tmpdir, experiment_name="test",
                      experiment_info=dict(name="Foo"), n_recent_kept=2, n_best_kept=2,
                      perf_func=lambda s: s['s'])
    cpman = cpman_f()
    for i, s in scores.items():
        cpman.save(dict(s=s), dict(i=i, s=s))

    for mode in ['resume', 'resume_or_start']:
        with pytest.raises(RuntimeError):
            cpman2 = cpman_f()
        cpman2: CheckpointManager = cpman_f(start_mode=mode)
        assert set(cpman2.saved) == set(map(str, kept))

    with pytest.raises(RuntimeError):
        cpman2.save(dict(s=10), dict(i=10, s=10))
    state, summary = cpman2.load_last()
    cpman2.save(dict(s=10), dict(i=10, s=10))

    cpman2.remove_old_checkpoints(0, 0)
    for mode in ['start', 'resume_or_start']:
        with pytest.raises(RuntimeError):
            cpman_f(start_mode='resume')
        cpman3: CheckpointManager = cpman_f(start_mode=mode)
        assert len(cpman3.saved) == 0

    cpman3.save(dict(s=8), dict(i=0, s=8))
    cpman4: CheckpointManager = cpman_f(start_mode='restart')
    assert len(cpman4.saved) == 0
