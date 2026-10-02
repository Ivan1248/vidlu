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
from vidlu.metrics import OutputKind

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


_OUTPUT_KIND_TO_CONVERTER = {"logits": ie.Predictions.from_logits,
                             "probs": ie.Predictions.from_probabilities}


def check_output_kind_storable(output_kind: OutputKind) -> None:
    """Raises `ValueError` unless the outputs are logits or probabilities.

    Hard (one-hot) outputs would store an unusable VLM response as a valid class 0.
    """
    if output_kind not in _OUTPUT_KIND_TO_CONVERTER:
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


def make_prediction_file_spec(
    dataset_info: T.Mapping[str, T.Any],
    method_name: str,
    method_seed: int | None = None,
    method_details: T.Mapping[str, T.Any] | None = None,
) -> PredictionFileSpec:
    """Builds the spec of a prediction file for a dataset split.

    Args:
        dataset_info: The `info` of an `irap_data.IRAPDataset` split of a known release.
        method_name, method_seed: See `irap_evaluation.MethodInfo`.
        method_details: JSON-compatible details of the run, stored with the code commit.

    Raises:
        ValueError: If the dataset is not a split of a known release.
    """
    if not is_irap_dataset(dataset_info):
        raise ValueError("Prediction files need a split of a known iRAP release, built by"
                         " irap_data.make_bh_data or make_vietnam_data.")
    method = ie.MethodInfo(name=method_name, seed=method_seed,
                           details={"code_commit": _get_code_commit(), **(method_details or {})})
    header = ie.PredictionHeader(dataset=dataset_info["dataset_name"], split=dataset_info["split"],
                                 method=method,
                                 context_offsets=tuple(dataset_info["context_offsets"]))
    return PredictionFileSpec(header, load_class_vocabulary(dataset_info["metadata_dir"]))


def write_output_predictions(
    path: str | Path,
    spec: PredictionFileSpec,
    segment_ids: T.Sequence[str],
    outputs: T.Sequence[np.ndarray],
    output_kind: OutputKind,
) -> None:
    """Writes model outputs as a prediction file.

    Args:
        outputs: Per-attribute (N, K_i) logits or probabilities, for all attributes of the
            dataset in its order.
    """
    check_output_kind_storable(output_kind)
    codes = spec.vocabulary.attribute_to_irap_codes
    from_outputs = _OUTPUT_KIND_TO_CONVERTER[output_kind]
    ie.write_predictions(path, from_outputs(spec.header, codes, segment_ids,
                                            dict(zip(codes, outputs, strict=True))))


def write_parsed_predictions(
    path: str | Path,
    spec: PredictionFileSpec,
    segment_to_predictions: T.Mapping[str, T.Mapping[str, AttributePrediction]],
    attributes: T.Sequence[str],
) -> None:
    """Writes parsed responses of `attributes` as hard predictions, an unusable one as invalid.

    Args:
        segment_to_predictions: Segment ID -> attribute -> parsed prediction, empty for a
            segment without a response.
    """
    codes = spec.vocabulary.restrict_to_attributes(attributes).attribute_to_irap_codes
    class_indices = attribute_predictions_to_class_indices(
        list(segment_to_predictions.values()), {a: len(c) for a, c in codes.items()})
    ie.write_predictions(path, ie.Predictions.from_class_indices(
        spec.header, codes, list(segment_to_predictions), class_indices))
