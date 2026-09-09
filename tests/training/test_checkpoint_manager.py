from functools import partial

import pytest

from vidlu.training.checkpoint_manager import CheckpointManager, find_checkpoint_dir


# `scores` maps the number of the training run to its performance. `kept` are the indices of the
# checkpoints that should survive: the 2 best (indices 1 and 2, with performances 4 and 3) and the
# 2 most recent (4 and 5). Checkpoint indices start at 0, so they are the keys of `scores` - 1.
@pytest.mark.parametrize("scores,kept", [({1: 2, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0}, {1, 2, 4, 5})])
def test_training_experiment(tmpdir, scores, kept):
    cpman_f = partial(CheckpointManager, checkpoints_root=tmpdir, experiment_name="test",
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
    state, summary, index = cpman2.load_last()
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


@pytest.mark.parametrize("scores", [{0: 2, 1: 4, 2: 3}])
def test_find_checkpoint_dir_agrees_with_the_checkpoint_manager(tmpdir, scores):
    """`find_checkpoint_dir` reads the saved files instead of instantiating a `CheckpointManager`,
    so it has to agree with it about which checkpoint is the best and which is the last."""
    cpman = CheckpointManager(checkpoints_root=tmpdir, experiment_name="test",
                              experiment_info=dict(name="Foo"), n_recent_kept=len(scores),
                              n_best_kept=len(scores), perf_func=lambda s: s['s'],
                              name_suffix_func=lambda s: f"{s['i']}_{s['s']:.3f}")
    for i, s in scores.items():
        cpman.save(dict(s=s), dict(i=i, s=s))

    assert find_checkpoint_dir(cpman.experiment_dir, "best") == cpman.best_checkpoint_path
    assert find_checkpoint_dir(cpman.experiment_dir, "last") == cpman.last_checkpoint_path
