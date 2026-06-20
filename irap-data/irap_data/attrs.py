"""
Canonical attribute subset definitions and helpers for IRAP GAIM experiments.
"""

from typing import Sequence


# Canonical iRAP attribute subset (per Kačan et al.). Used as the cross-dataset
# default; per-dataset availability is narrowed via filter_attrs_with_values.
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
    return ATTRS_TO_INCLUDE


def filter_attrs_with_values(
    attrs: Sequence[str],
    attr_to_value_to_class_idx: dict[str, dict[str, int]],
) -> tuple[str, ...]:
    """Keep only attributes that carry a non-empty value vocabulary in the dataset.

    Drops attributes that are absent from the dataset's metadata or have no values
    (e.g. IRAP-Vietnam's flow attributes, empty in every coding table). Used as the
    single source of truth for the attribute subset so the VLM data wrapper and the
    metrics never disagree. For IRAP-BH every canonical attribute has values, so this
    is a no-op there.
    """
    return tuple(a for a in attrs if attr_to_value_to_class_idx.get(a))

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
        List of attribute indices.
    """
    attr_name_to_idx = {name: idx for idx, name in enumerate(dataset_attribute_names)}
    return [attr_name_to_idx[name] for name in attr_names]
