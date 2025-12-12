"""
Canonical attribute subset definitions and helpers for IRAP GAIM experiments.
"""
from __future__ import annotations

from typing import Sequence


# IRAP-BH attribute subset
ATTRS_TO_INCLUDE = (
    "Bicycle observed flow",
    "Pedestrian observed flow across the road",
    "Pedestrian observed flow along the road driver-side",
    "Pedestrian observed flow along the road passenger-side",
    "Speed management / traffic calming",
    "Number of lanes",
    "Lane width",
    "Curvature",
    "Quality of curve",
    "Upgrade cost",
    "Median Type",
    "Skid resistance / grip",
    "Road condition",
    "Vehicle parking",
    "Grade",
    "Roadworks",
    "Sight distance",
    "Delineation",
    "Street lighting",
    "Service road",
    "Roadside severity - driver-side distance",
    "Roadside severity - driver-side object",
    "Roadside severity - passenger-side distance",
    "Roadside severity - passenger-side object",
    "Paved shoulder - driver-side",
    "Paved shoulder - passenger-side",
    "Intersection type",
    "Intersection channelisation",
    "Intersection quality",
    "Property access points",
    "Land use - driver-side",
    "Land use - passenger-side",
    "Area type",
    "Pedestrian crossing - inspected road",
    "Pedestrian crossing quality",
    "Pedestrian crossing - side road",
    "Sidewalk - driver-side",
    "Sidewalk - passanger-side",
    "Bicycle facility",
    "School zone warning",
    "School zone crossing supervisor",
)


def get_attrs_to_include() -> tuple[str, ...]:
    """Returns the canonical attribute subset used in the paper experiments."""
    return ATTRS_TO_INCLUDE


def map_attr_names_to_indices(
    attr_names: Sequence[str],
    dataset_attribute_names: Sequence[str],
) -> list[int] | None:
    """
    Maps a sequence of attribute names to their indices in the dataset's attribute order.
    
    Args:
        attr_names: Sequence of attribute names to include.
        dataset_attribute_names: Ordered sequence of all attribute names from the dataset.
    
    Returns:
        List of attribute indices, or None if attr_names is None/empty (meaning include all).
    """
    if attr_names is None or len(attr_names) == 0:
        return None
    
    attr_name_to_idx = {name: idx for idx, name in enumerate(dataset_attribute_names)}
    indices = []
    for name in attr_names:
        if name not in attr_name_to_idx:
            raise ValueError(
                f"Attribute '{name}' not found in dataset attributes. "
                f"Available: {list(dataset_attribute_names)[:5]}..."
            )
        indices.append(attr_name_to_idx[name])
    return indices


