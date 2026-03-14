"""
Report the most common value for each attribute from the iRAP-BH training set.

Outputs a JSON file mapping attribute names to their most frequent value.

Usage:
    IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.attribute_most_common_report --output most_common_values.json
"""

import argparse
import json
from pathlib import Path

from vidlu_irap_gaim.data.attribute_frequencies import compute_attribute_frequency_stats
from vidlu_irap_gaim.data import make_bih_data
from vidlu_irap_gaim.vlm.response_parser import build_idx_to_value


def main():
    parser = argparse.ArgumentParser(
        description="Report most common attribute values from iRAP-BH training set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("most_common_values.json"),
        help="Output JSON file path (default: most_common_values.json)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to analyze (default: train)",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    datasets = make_bih_data()
    dataset = datasets[args.split]
    print(f"Samples: {len(dataset)}")

    print("Computing attribute frequencies...")
    stats = compute_attribute_frequency_stats(dataset)

    attr_to_value_to_class_idx = dataset.info.attr_to_value_to_class_idx
    attr_names = list(attr_to_value_to_class_idx.keys())
    idx_to_value = build_idx_to_value(attr_to_value_to_class_idx)

    most_common_values = {
        attr_names[i]: idx_to_value[attr_names[i]][stats.most_common_class_indices[i]]
        for i in range(len(attr_names))
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(most_common_values, f, indent=2)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
