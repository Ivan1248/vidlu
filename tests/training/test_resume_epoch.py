"""Regression tests for resuming training (`run.py ... -r`) at the correct epoch.

These target the epoch-continuation contract directly, without a GPU, model, or
dataset:

- ``EpochLoop`` (`vidlu/training/trainers.py`) holds the epoch counter, serializes
  it via ``state_dict``/``load_state_dict``, and on ``run(restart=False)`` must
  continue from ``state.epoch + 1`` rather than restarting from 0.
- ``get_report_iters`` (`vidlu/experiments.py`) selects which epochs get a
  checkpoint. Resume reads the epoch back from whichever checkpoint was saved, so
  the result is independent of ``eval_count``; the only requirement is that the
  final epoch is always checkpointed (so a completed run is recognized as done).
"""

import pytest

from vidlu.training.trainers import EpochLoop
from vidlu.experiments import get_report_iters


def _noop_step(loop, batch):
    return None


def _make_loop():
    """An EpochLoop whose completed-epoch indices are recorded in ``executed``."""
    loop = EpochLoop(_noop_step)
    executed = []
    loop.epoch_completed.add_handler(lambda state: executed.append(state.epoch))
    return loop, executed


DATA = [0, 1, 2]  # 3 "batches"; only len() and iteration matter here


@pytest.mark.parametrize("saved_epoch,epoch_count", [(0, 2), (0, 3), (1, 5), (2, 10)])
def test_resume_continues_at_next_epoch(saved_epoch, epoch_count):
    # First run: stop right after `saved_epoch` (simulated by capping max_epochs).
    loop1, executed1 = _make_loop()
    loop1.run(DATA, max_epochs=saved_epoch + 1, restart=True)
    assert executed1 == list(range(saved_epoch + 1))
    assert loop1.state.epoch == saved_epoch

    # Resume: a fresh loop loads the saved state and continues without restarting.
    state_dict = loop1.state_dict()
    assert state_dict == {"epoch": saved_epoch}

    loop2, executed2 = _make_loop()
    loop2.load_state_dict(state_dict)
    assert loop2.state.epoch == saved_epoch  # restored, not reset to -1

    loop2.run(DATA, max_epochs=epoch_count, restart=False)

    # It must run exactly the remaining epochs, contiguous and correctly numbered.
    assert executed2 == list(range(saved_epoch + 1, epoch_count))
    assert loop2.state.epoch == epoch_count - 1


def test_resume_of_completed_run_is_noop():
    epoch_count = 4
    loop1, _ = _make_loop()
    loop1.run(DATA, max_epochs=epoch_count, restart=True)
    assert loop1.state.epoch == epoch_count - 1  # fully finished

    loop2, executed2 = _make_loop()
    loop2.load_state_dict(loop1.state_dict())
    with pytest.warns(UserWarning, match="already completed"):
        loop2.run(DATA, max_epochs=epoch_count, restart=False)

    assert executed2 == []  # nothing rerun
    assert loop2.state.epoch == epoch_count - 1


@pytest.mark.parametrize("epoch_count", [1, 4, 10])
@pytest.mark.parametrize("eval_count", [1, 3, 5, 10, 13])
def test_final_epoch_is_always_checkpointed(eval_count, epoch_count):
    """Resume works for any eval_count because the last epoch is always saved.

    Whatever epoch the last checkpoint holds, resume continues from epoch+1 (see
    ``test_resume_continues_at_next_epoch``), so the only eval_count-dependent
    requirement is that the final epoch produces a checkpoint; otherwise a
    finished run could not be recognized as complete.
    """
    save_epochs = get_report_iters(eval_count, epoch_count)
    assert save_epochs, "at least one epoch must be checkpointed"
    assert all(0 <= e < epoch_count for e in save_epochs)
    assert max(save_epochs) == epoch_count - 1
