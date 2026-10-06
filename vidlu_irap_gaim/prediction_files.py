"""Writing `irap_evaluation` prediction files, which `irap-eval` scores, ensembles and exports
without rerunning the model.
"""

import dataclasses as dc
import subprocess
import typing as T
from pathlib import Path

import numpy as np

import irap_evaluation as ie
from irap_data import ClassVocabulary, load_class_vocabulary
from vidlu.experiments import TrainingExperiment, get_method_string
from vidlu.metrics import OutputKind
from vidlu.training import get_training_data

from vidlu_irap_gaim.vlm.predictions import attribute_predictions_to_class_indices
from vidlu_irap_gaim.vlm.response_parser import AttributePrediction


def _get_code_commit() -> str | None:
    """The commit of the vidlu checkout, with `-dirty` if it has uncommitted changes."""
    try:
        completed = subprocess.run(["git", "describe", "--always", "--dirty"],
                                   cwd=Path(__file__).resolve().parent, capture_output=True,
                                   text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


_OUTPUT_KIND_TO_PREDICTIONS_FACTORY = {"logits": ie.Predictions.from_logits,
                             "probs": ie.Predictions.from_probabilities}


def check_output_kind_storable(output_kind: OutputKind) -> None:
    if output_kind not in _OUTPUT_KIND_TO_PREDICTIONS_FACTORY:
        raise ValueError(f"Prediction files need logits or probabilities, not {output_kind!r}"
                         f" outputs. Write parsed VLM responses with write_parsed_predictions.")


@dc.dataclass(frozen=True)
class PredictionFileSpec:
    """What a prediction file of a dataset split holds besides the predictions."""

    header: ie.PredictionHeader
    vocabulary: ClassVocabulary  # all attributes of the dataset, in its order


def is_irap_dataset(dataset_info: T.Mapping[str, T.Any]) -> bool:
    """Whether the dataset is a split of a known iRAP release, which prediction files need."""
    return dataset_info.get("dataset_name") is not None


def make_prediction_file_spec(dataset_info: T.Mapping[str, T.Any],
                              model_info: ie.ModelInfo) -> PredictionFileSpec:
    """Builds the spec of a prediction file for a dataset split.

    Args:
        dataset_info: The `info` of an `irap_data.IRAPDataset` split of a known release.
        model_info: The model that makes the predictions. The code commit is added to its
            details under 'commit'.
    """
    if not is_irap_dataset(dataset_info):
        raise ValueError("Prediction files need a split of a known iRAP release, built by"
                         " irap_data.make_bh_data or make_vietnam_data.")
    model_info = dc.replace(model_info,
                            details={"commit": _get_code_commit(), **(model_info.details or {})})
    header = ie.PredictionHeader(dataset=dataset_info["dataset_name"], split=dataset_info["split"],
                                 model=model_info,
                                 context_offsets=tuple(dataset_info["context_offsets"]))
    return PredictionFileSpec(header, load_class_vocabulary(dataset_info["metadata_dir"]))


def _get_parent_datasets(dataset) -> T.Sequence | None:
    """The datasets that `dataset` is derived from, or `None` if it is not derived from datasets."""
    if (parts := getattr(dataset, "datasets", None)) is not None:
        return parts
    data = getattr(dataset, "data", None)
    return (data,) if hasattr(data, "info") or hasattr(data, "datasets") else None


def _get_split_names(data_key: str, dataset) -> tuple[str, ...]:
    """The splits of known iRAP releases that the examples of the experiment data entry
    `data_key` may come from, in order of appearance.

    The splits are taken from the `info` of the datasets that `dataset` is derived from (see
    `_get_parent_datasets`). The `info` of a derived dataset is not used, since a join keeps only
    the `info` of its first dataset. A subset of a join is taken to come from all joined splits.

    Raises:
        ValueError: If `dataset` is derived from a dataset that is not a split of a known iRAP
            release, e.g. `PseudoLabeledDataset` or `ZipDataset`.
    """
    if (parents := _get_parent_datasets(dataset)) is not None:
        return tuple(dict.fromkeys(split for parent in parents
                                   for split in _get_split_names(data_key, parent)))
    info = getattr(dataset, "info", None)
    if info is None or not is_irap_dataset(info):
        raise ValueError(f"The data entry {data_key!r} is derived from a {type(dataset).__name__}"
                         f" that is not a split of a known iRAP release, so the prediction file"
                         f" cannot name its split.")
    return (info["split"],)


def make_model_info_from_experiment(experiment: TrainingExperiment,
                                    method_name: str | None = None,
                                    seed: int | None = None) -> ie.ModelInfo:
    """The `irap_evaluation.ModelInfo` of the experiment's model as constructed, e.g. as loaded by
    `run.py test`, with the experiment directory in its details under 'checkpoint'.

    Args:
        method_name: Defaults to the experiment's trainer and model factory strings (see
            `vidlu.experiments.get_method_string`), followed by '_best' if the best checkpoint
            was loaded (`run.py -r best`), so that the best and the last checkpoint of a run
            are models of different methods with differently named prediction files.
        seed: The model's seed, e.g. its training seed (`run.py train -s`), which the experiment
            does not store.
    """
    training_splits, early_stopping_splits = _get_training_split_names(experiment)
    return ie.ModelInfo(
        method_name=method_name or _get_default_method_name(experiment),
        training_splits=training_splits, early_stopping_splits=early_stopping_splits, seed=seed,
        details={"checkpoint": str(experiment.cpman.experiment_dir)})


def _get_default_method_name(experiment: TrainingExperiment) -> str:
    """See `make_model_info_from_experiment`."""
    name = get_method_string(experiment.training_args, include_model=True)
    return f"{name}_best" if experiment.checkpoint_resumption == "best" else name


def _get_training_split_names(
    experiment: TrainingExperiment,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """The `training_splits` and `early_stopping_splits` (see `irap_evaluation.ModelInfo`) of the
    experiment's model as constructed.

    A split that ranks the checkpoints and that the model was also fitted on is a training split,
    not an early stopping split.
    """
    if experiment.checkpoint_resumption is None:
        # Initial parameters were fitted on nothing in the data, or on data that is not known.
        return (() if experiment.training_args.params is None else None), ()
    data = experiment.data
    training_splits = tuple(dict.fromkeys(  # e.g. a labeled subset and the rest of "train"
        split for key, ds in get_training_data(data).items()
        for split in _get_split_names(key, ds)))
    if experiment.checkpoint_resumption == "last":
        return training_splits, ()
    if (key := experiment.checkpoint_val_data_key) is None:
        raise ValueError("The best checkpoint was loaded, but no data entry ranks checkpoints.")
    return training_splits, tuple(s for s in _get_split_names(key, data[key])
                                  if s not in training_splits)


def _write_predictions_to_dir(output_dir: str | Path, predictions: ie.Predictions,
                              header: ie.PredictionHeader) -> Path:
    """Writes `predictions` to a file in `output_dir` named by
    `irap_evaluation.make_prediction_file_name`."""
    path = Path(output_dir) / ie.make_prediction_file_name(header.model.method_name,
                                                            header.model.seed, header.split)
    ie.write_predictions(path, predictions)
    return path


def write_output_predictions(
    output_dir: str | Path,
    spec: PredictionFileSpec,
    segment_ids: T.Sequence[str],
    outputs: T.Sequence[np.ndarray],
    output_kind: OutputKind,
) -> Path:
    """Writes model outputs to a prediction file in `output_dir` named by
    `irap_evaluation.make_prediction_file_name`.

    Args:
        outputs: Per-attribute (N, K_i) logits or probabilities, for all attributes of the
            dataset in its order.
    """
    check_output_kind_storable(output_kind)
    codes = spec.vocabulary.attribute_to_irap_codes
    make_predictions = _OUTPUT_KIND_TO_PREDICTIONS_FACTORY[output_kind]
    return _write_predictions_to_dir(
        output_dir, make_predictions(spec.header, codes, segment_ids,
                                 dict(zip(codes, outputs, strict=True))), spec.header)


def write_parsed_predictions(
    output_dir: str | Path,
    spec: PredictionFileSpec,
    segment_to_predictions: T.Mapping[str, T.Mapping[str, AttributePrediction]],
    attributes: T.Sequence[str],
) -> Path:
    """Writes parsed responses of `attributes` as hard predictions, an unusable one as invalid,
    to a prediction file in `output_dir` named by `irap_evaluation.make_prediction_file_name`.

    Args:
        segment_to_predictions: Segment ID -> attribute -> parsed prediction, empty for a
            segment without a response.
    """
    codes = spec.vocabulary.restrict_to_attributes(attributes).attribute_to_irap_codes
    class_indices = attribute_predictions_to_class_indices(
        list(segment_to_predictions.values()), {a: len(c) for a, c in codes.items()})
    return _write_predictions_to_dir(
        output_dir, ie.Predictions.from_class_indices(
            spec.header, codes, list(segment_to_predictions), class_indices), spec.header)
