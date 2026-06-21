"""Tests for `vidlu.utils.logger.Logger`.

The Logger is a single, append-only, serializable transcript shared by reference
across training callbacks. These tests pin the contract that makes resume work:
messages from every caller accumulate in one record list, the state round-trips
through a checkpoint, and `print_all` replays it on a fresh instance.
"""

from vidlu.utils.logger import Logger


def _silent():
    """A Logger that records but doesn't print (keeps test output clean)."""
    return Logger(emit=lambda _: None)


def test_shared_logger_accumulates_messages_from_multiple_callers():
    """A logger passed by reference collects every caller's messages."""
    logger = _silent()
    a, b = logger, logger  # mimics callbacks all holding the same instance
    a.log("training report")
    b.log("eval report")
    text = logger.as_text()
    assert "training report" in text and "eval report" in text


def test_state_dict_round_trips_records():
    logger = _silent()
    logger.log("epoch 1 val")
    logger.log("epoch 2 val")

    restored = _silent()
    restored.load_state_dict(logger.state_dict())
    assert restored.records == logger.records


def test_load_state_dict_then_print_all_replays_on_fresh_instance(capsys):
    """The resume path: a brand-new logger replays the saved transcript to stdout."""
    saved = _silent()
    saved.log("from previous run")

    fresh = Logger()  # default emit -> tqdm.write -> stdout
    fresh.load_state_dict(saved.state_dict())
    fresh.print_all()
    assert "from previous run" in capsys.readouterr().out


def test_continued_logging_after_resume_is_recorded():
    """After a resume, new messages append to the restored transcript."""
    saved = _silent()
    saved.log("from previous run")

    resumed = _silent()
    resumed.load_state_dict(saved.state_dict())
    resumed.log("from resumed run")

    text = resumed.as_text()
    assert "from previous run" in text and "from resumed run" in text


def test_as_text_matches_logged_lines():
    logger = _silent()
    logger.log("a")
    logger.log("b")
    lines = logger.as_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("a") and lines[1].endswith("b")


def test_state_dict_is_plain_data():
    """No callables in the serialized state (checkpoints must stay portable)."""
    logger = _silent()
    logger.log("x")
    state = logger.state_dict()
    assert set(state) == {"records"}
    assert all(isinstance(r, tuple) and all(isinstance(s, str) for s in r)
               for r in state["records"])


def test_load_state_dict_does_not_alias_source_records():
    """Loading copies the records so the two loggers don't share a list."""
    source = _silent()
    source.log("x")
    other = _silent()
    other.load_state_dict(source.state_dict())
    other.log("y")
    assert "y" not in source.as_text()
