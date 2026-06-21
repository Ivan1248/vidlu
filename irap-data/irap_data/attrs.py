"""
Canonical attribute subset definitions and helpers for IRAP GAIM experiments.
"""

from typing import Sequence


# Canonical iRAP attribute subset (per Kačan et al.). Used as the cross-dataset
# default. Per-dataset availability is narrowed via filter_labeled_attrs.
IRAP_BH_ATTRS_TO_INCLUDE = (
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

# IRAP-Vietnam attributes that are present in iRAP-BH.
IRAP_VIETNAM_ATTRS_SHARED = (
    "Speed management / traffic calming",
    "Number of lanes",
    "Lane width",
    "Curvature",
    "Quality of curve",
    "Median Type",
    "Skid resistance / grip",
    "Road condition",
    "Vehicle parking",
    "Grade",
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
    "School zone warning",
    "School zone crossing supervisor",
)

# All IRAP-Vietnam attributes: the shared subset plus attributes not present in iRAP-BH.
IRAP_VIETNAM_ATTRS_ALL = IRAP_VIETNAM_ATTRS_SHARED + (
    "Carriageway label",
    "Speed limit",
    "Shoulder rumble strips",
    "Intersecting road volume",
)

def get_attrs_to_include() -> tuple[str, ...]:
    return IRAP_BH_ATTRS_TO_INCLUDE


def filter_labeled_attrs(
    attrs: Sequence[str],
    attr_to_num_labeled: dict[str, int],
) -> tuple[str, ...]:
    """Keep only attributes with >0 labeled (non-ignore) examples."""
    return tuple(a for a in attrs if attr_to_num_labeled.get(a, 0) > 0)


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
