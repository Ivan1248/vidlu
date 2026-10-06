"""The splits that `prediction_files.make_model_info_from_experiment` records, with a stub
experiment, so these run without IRAP_HOME."""

from types import SimpleNamespace

import pytest

from vidlu_irap_gaim.prediction_files import make_model_info_from_experiment


def _split(split):
    return SimpleNamespace(info={"dataset_name": "bh", "split": split})


def _make_experiment(checkpoint_resumption, checkpoint_val_data_key="val", **data):
    data = data or dict(train=_split("train"), val=_split("val"), test=_split("test"))
    return SimpleNamespace(data=data, checkpoint_resumption=checkpoint_resumption,
                           checkpoint_val_data_key=checkpoint_val_data_key,
                           training_args=SimpleNamespace(params=None),
                           cpman=SimpleNamespace(experiment_dir="experiment"))


def test_the_ranking_split_of_the_best_checkpoint_is_an_early_stopping_split():
    model_info = make_model_info_from_experiment(_make_experiment("best"), "m")
    assert model_info.training_splits == ("train",)
    assert model_info.early_stopping_splits == ("val",)


def test_the_last_checkpoint_has_no_early_stopping_split():
    model_info = make_model_info_from_experiment(_make_experiment("last"), "m")
    assert model_info.training_splits == ("train",)
    assert model_info.early_stopping_splits == ()


def test_a_ranking_split_that_is_also_trained_on_is_only_a_training_split():
    experiment = _make_experiment("best", train=_split("val"), val=_split("val"))
    model_info = make_model_info_from_experiment(experiment, "m")
    assert model_info.training_splits == ("val",)
    assert model_info.early_stopping_splits == ()


def test_the_best_checkpoint_needs_a_ranking_entry():
    with pytest.raises(ValueError):
        make_model_info_from_experiment(_make_experiment("best", checkpoint_val_data_key=None),
                                        "m")
