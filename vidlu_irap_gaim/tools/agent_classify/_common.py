"""Shared helpers for the agent-based iRAP classification tools.

``prepare_agent_tasks`` and ``evaluate_agent_predictions`` must load the dataset
and derive the attribute subset identically, so the prompt the agent receives and
the metrics it is scored against cover exactly the same attributes. This module is
the single place that pairing lives.
"""

from pathlib import Path

from irap_data import make_irap_data_by_name
from irap_data.attrs import get_attrs_to_include, filter_attrs_with_values


def agent_artifact_paths(output_dir: Path | str, dataset_name: str, split: str) -> dict[str, Path]:
    """Conventional artifact file names for one ``(dataset, split)`` run.

    The single source of truth for the file names shared across the workflow:
    ``prepare`` writes ``tasks`` / ``prompt`` and tells the agent to write
    ``predictions``; ``evaluate`` reads ``predictions`` back. Keeping the names
    here means the two halves can never drift.
    """
    output_dir = Path(output_dir)
    stem = f"{dataset_name}_{split}"
    return {
        "tasks": output_dir / f"agent_tasks_{stem}.json",
        "prompt": output_dir / f"agent_prompt_{stem}.md",
        "predictions": output_dir / f"agent_predictions_{stem}.json",  # agent writes here
    }


def load_split_and_attrs(dataset_name: str, split: str):
    """Load an IRAP split and the attribute subset to classify for it.

    The subset comes from the dataset's own metadata via ``filter_attrs_with_values``
    — the single source of truth shared with ``get_irap_metrics`` and the VLM data
    factories. It drops attributes with no value vocabulary (e.g. IRAP-Vietnam's
    empty flow attributes); a no-op for IRAP-BH.

    Args:
        dataset_name: ``"bih"`` or ``"vietnam"``.
        split: Dataset split (``"train"``, ``"val"``, ``"test"``).

    Returns:
        Tuple of ``(dataset, attr_to_value_to_class_idx, attrs_to_include)``.
    """
    dataset = make_irap_data_by_name(dataset_name)[split]
    attr_to_value_to_class_idx = dataset.info.attr_to_value_to_class_idx
    attrs_to_include = list(
        filter_attrs_with_values(get_attrs_to_include(), attr_to_value_to_class_idx)
    )
    return dataset, attr_to_value_to_class_idx, attrs_to_include
