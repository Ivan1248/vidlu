"""
Report the number of segments in each split of the iRAP datasets.

Loads each release (iRAP-BH, iRAP-Vietnam) via its factory and prints a table of
per-split sizes plus per-dataset totals. Sizes reflect the same options used in
training (e.g. BiH's N-context filter is on by default), so the numbers match the
`n` reported by the metrics. Only metadata is read -- no images are loaded.

Run as a plain file (not `-m`), so importing the `vidlu_irap_gaim` package -- which
pulls in `irap_data` -- is avoided; this script imports `irap_data` directly and
adds it to `sys.path` itself (matching `scripts/_context.py`), so no install or
`run.py` bootstrap is needed:

    IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/dataset_split_sizes.py
    IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/dataset_split_sizes.py --markdown
    IRAP_HOME=~/data/datasets python vidlu_irap_gaim/tools/dataset_split_sizes.py --datasets bih --json sizes.json
"""

import argparse
import json
import sys
from pathlib import Path

# `irap_data` is not installed; it lives at <repo>/irap-data (src-layout) and is
# normally put on sys.path by scripts/_context.py. Do the same here so this tool
# runs standalone. Layout: <repo>/vidlu_irap_gaim/tools/<this file>.
_irap_data_project = Path(__file__).resolve().parents[2] / "irap-data"
if _irap_data_project.is_dir() and str(_irap_data_project) not in sys.path:
    sys.path.insert(0, str(_irap_data_project))

from irap_data import IRAP_DATASET_FACTORIES, make_irap_data_by_name  # noqa: E402

# Human-readable labels for the registry keys used in the report headings.
DISPLAY_NAMES = {"bih": "iRAP-BH", "vietnam": "iRAP-Vietnam"}


def collect_split_sizes(names):
    """Return {dataset_name: {split_name: size}} for the requested releases.

    A release that fails to load (e.g. its path is not present under IRAP_HOME)
    is reported and skipped rather than aborting the whole run.
    """
    sizes = {}
    for name in names:
        try:
            data = make_irap_data_by_name(name)
        except Exception as e:  # noqa: BLE001 - report and continue past a missing release
            print(f"[warning] could not load '{name}': {type(e).__name__}: {e}")
            continue
        # `data` is dict-like (split name -> IRAPDataset); iterate the splits
        # actually present (labeled train/val/test, plus any unlabeled_* subsets).
        sizes[name] = {split: len(data[split]) for split in data}
    return sizes


def print_table(sizes):
    """Print an aligned per-split, per-dataset table with totals."""
    for name, split_sizes in sizes.items():
        title = DISPLAY_NAMES.get(name, name)
        width = max((len(s) for s in split_sizes), default=5)
        print(f"\n{title}")
        print(f"  {'Split'.ljust(width)}  {'Segments':>10}")
        print(f"  {'-' * width}  {'-' * 10}")
        for split, n in split_sizes.items():
            print(f"  {split.ljust(width)}  {n:>10,}")
        print(f"  {'total'.ljust(width)}  {sum(split_sizes.values()):>10,}")


def print_markdown(sizes):
    """Print a Markdown table (rows = dataset x split) for pasting into notes."""
    print("\n| Dataset | Split | Segments |")
    print("|---|---|---:|")
    for name, split_sizes in sizes.items():
        title = DISPLAY_NAMES.get(name, name)
        for split, n in split_sizes.items():
            print(f"| {title} | {split} | {n:,} |")
        print(f"| {title} | **total** | **{sum(split_sizes.values()):,}** |")


def main():
    parser = argparse.ArgumentParser(
        description="Report per-split sizes of the iRAP datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(IRAP_DATASET_FACTORIES),
        default=list(IRAP_DATASET_FACTORIES),
        help="Releases to report (default: all).",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Also print a Markdown table."
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path to write sizes as JSON."
    )
    args = parser.parse_args()

    sizes = collect_split_sizes(args.datasets)
    if not sizes:
        parser.error("no datasets could be loaded; check IRAP_HOME and paths.")

    print_table(sizes)
    if args.markdown:
        print_markdown(sizes)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(sizes, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
