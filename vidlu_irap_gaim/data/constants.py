RGB_MEAN: tuple[float, float, float] = (0.53354913, 0.52727484, 0.48752149)
RGB_STD: tuple[float, float, float] = (0.20401913, 0.20417478, 0.25402164)
INPUT_DIM_RGB: tuple[int, int, int] = (384, 288, 3)

# Ignore index for cross-entropy when an attribute is missing/unmappable.
IGNORE_LABEL_INDEX: int = -1


class MetaFiles:
    """File names (relative to a dataset's metadata directory)."""

    ATTRIBUTE_METADATA = "attribute_metadata.json"
    SPLITS = "splits.json"
    SEGMENT_ID_TO_DATA_PATHS = "segment_id_to_data_paths_rel.json"
    SEGMENT_ID_TO_ROAD_DATA = "segment_id_to_road_data.json"
    ROAD_ID_TO_SEGMENT_ID_SEQUENCE = "road_id_to_segment_id_sequence.json"
