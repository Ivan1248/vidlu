import pytest

from vidlu.experiments import ValidationCheckpointHandler, get_checkpoint_val_data_key


def test_the_first_validation_entry_ranks_checkpoints():
    data = {"train": [], "val_quick": [], "val": [], "test": []}
    assert get_checkpoint_val_data_key(data) == "val_quick"


def test_no_validation_entry_ranks_no_checkpoints():
    assert get_checkpoint_val_data_key({"train": [], "test": []}) is None


@pytest.mark.parametrize("checkpoint_val_data_key", ["val_x", "test"])
def test_the_handler_refuses_a_key_of_no_validation_entry(checkpoint_val_data_key):
    with pytest.raises(ValueError):
        ValidationCheckpointHandler(
            data={"train": [], "val": [], "test": []}, cpman=None, main_metrics=[],
            eval_count=1, epoch_count=1, logger=None,
            checkpoint_val_data_key=checkpoint_val_data_key)
