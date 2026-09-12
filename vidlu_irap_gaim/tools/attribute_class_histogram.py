"""Plots class frequency distributions grouped by attribute.

Generates a grouped horizontal bar chart depicting class counts or normalized shares
per attribute. Run as a plain file, like `attribute_distribution_report.py`, whose counting
it reuses.
"""

import argparse
import sys
import typing as T
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# `irap_data` lives at <repo>/irap-data (src-layout) and is normally put on sys.path by
# scripts/_context.py; see attribute_distribution_report for the rationale.
_irap_data_project = Path(__file__).resolve().parents[2] / "irap-data"
if _irap_data_project.is_dir() and str(_irap_data_project) not in sys.path:
    sys.path.insert(0, str(_irap_data_project))

from attribute_distribution_report import (ATTRIBUTE_ORDERS, CLASS_ORDERS,  # noqa: E402
                                           AttributeDistribution, annotated,
                                           compute_attribute_distributions, has_labels,
                                           sort_distributions)
from irap_data import IRAP_DATASET_FACTORIES, make_irap_data_by_name  # noqa: E402

#: Alternating group shades, so where one attribute ends and the next begins is readable
#: without relying on colour vision.
_GROUP_COLORS = ("#4878a8", "#a0c4e4")
_ABSENT_COLOR = "#c44e52"

#: Horizontal room reserved for the attribute names, left of the class tick labels.
#: Class names are drawn at 6pt, so this clears roughly 25 characters of them.
_LABEL_GUTTER_PT = 76.0


def _bar_positions(distributions: list[AttributeDistribution]) -> tuple[np.ndarray, np.ndarray]:
    """Y positions of the class bars, one unit apart with a one-unit gap between attributes.

    Returns:
        `(positions, group_starts)`: a position per class bar, and the position of each
        attribute's first bar.
    """
    sizes = np.array([d.num_classes for d in distributions])
    group_starts = np.concatenate([[0], np.cumsum(sizes[:-1] + 1)])
    positions = np.concatenate([start + np.arange(size)
                                for start, size in zip(group_starts, sizes)])
    return positions, group_starts


def plot_class_frequencies(distributions: list[AttributeDistribution], *, title: str,
                           normalize: bool = False, log: bool = True):
    """Plots class frequency distributions grouped by attribute.

    Args:
        distributions: Sequence of `AttributeDistribution` objects to plot.
        title: Figure title text.
        normalize: If True, plots frequencies normalized by total labeled count per attribute.
        log: If True, uses logarithmic horizontal scaling for frequencies.

    Returns:
        Matplotlib `Figure` object containing the generated plot.

    Raises:
        ValueError: If no annotated attribute distributions are provided.
    """
    distributions = annotated(distributions)
    if not distributions:
        raise ValueError("No annotated attributes to plot.")

    positions, group_starts = _bar_positions(distributions)
    values, labels, colors = [], [], []
    for group_index, d in enumerate(distributions):
        scale = d.num_labeled if normalize else 1
        for class_name, count in zip(d.class_names, d.counts):
            values.append(count / scale)
            labels.append(class_name)
            colors.append(_ABSENT_COLOR if count == 0
                          else _GROUP_COLORS[group_index % len(_GROUP_COLORS)])
    values = np.asarray(values, dtype=float)

    # A class with no example needs a visible bar on either scale, or it reads as "not
    # plotted" rather than "never observed" – and those are exactly the classes worth
    # finding. A log axis cannot represent 0 at all; a linear axis represents it as zero
    # width, which is just as invisible. So both get a small floor, coloured distinctly
    # and explained in the legend.
    positive = values[values > 0]
    if not len(positive):
        floor = 0.0
    elif log:
        floor = positive.min() / 4
    else:
        floor = values.max() / 200
    drawn = np.where(values > 0, values, floor)

    height = max(4.0, 0.16 * len(values) + 2.0)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(positions, drawn, height=0.8, color=colors)
    if log:
        ax.set_xscale("log")
        if floor > 0:
            ax.set_xlim(left=floor / 3)

    ax.set_yticks(positions, labels, fontsize=6)
    ax.invert_yaxis()  # first attribute at the top, reading order
    ax.set_xlabel("share of the attribute's labelled examples" if normalize
                  else "number of examples")
    ax.set_title(title)
    ax.grid(axis="x", which="both", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # Attribute names go in the left margin, outside the class tick labels. The offset is
    # in points from the axis rather than an axes fraction, so it stays clear of the tick
    # labels regardless of figure size; the margin below is sized to match.
    for start, d in zip(group_starts, distributions):
        ax.annotate(d.name, xy=(0, start + (d.num_classes - 1) / 2),
                    xycoords=ax.get_yaxis_transform(), xytext=(-_LABEL_GUTTER_PT, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, fontweight="bold")
    for start, d in zip(group_starts[:-1], distributions[:-1]):  # separators between groups
        ax.axhline(start + d.num_classes - 0.1, color="0.85", linewidth=0.6)

    if (values == 0).any():
        # A proxy patch: an empty `barh` contributes no artist for the legend to colour.
        ax.legend(handles=[Patch(color=_ABSENT_COLOR, label="never observed (0 examples)")],
                  loc="lower right", fontsize=7)

    fig.subplots_adjust(left=0.34, right=0.98, top=1 - 0.55 / height, bottom=0.55 / height)
    return fig


def main(argv: T.Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-class frequencies of all attributes of an iRAP release.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--release", default="vietnam", choices=list(IRAP_DATASET_FACTORIES),
                        help="Release to plot (default: vietnam).")
    parser.add_argument("--split", default="train", help="Split to plot (default: train).")
    parser.add_argument("--normalize", action="store_true",
                        help="Plot shares within each attribute instead of raw counts,"
                             " making attributes comparable.")
    parser.add_argument("--linear", action="store_true",
                        help="Use a linear value axis (default: logarithmic, since counts"
                             " span several orders of magnitude).")
    parser.add_argument("--sort-attributes", choices=ATTRIBUTE_ORDERS, default="schema",
                        help="Order of the attribute groups: 'schema' keeps the release's"
                             " attribute order, 'count' sorts by descending labelled"
                             " examples, 'imbalance' puts the most skewed first"
                             " (default: schema).")
    parser.add_argument("--sort-classes", choices=CLASS_ORDERS, default="schema",
                        help="Order of the classes within each attribute: 'schema' keeps"
                             " class-index order, which is meaningful for ordinal"
                             " attributes such as speed limits; 'count' sorts by descending"
                             " examples, as for --sort-attributes (default: schema).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Image path (default: <release>_<split>_class_frequencies.png)."
                             " The extension selects the format; .pdf gives vector output.")
    parser.add_argument("--dpi", type=int, default=200, help="Raster resolution (default: 200).")
    args = parser.parse_args(argv)

    data = make_irap_data_by_name(args.release)
    if args.split not in data:
        parser.error(f"release '{args.release}' has no split '{args.split}';"
                     f" available: {', '.join(data)}")
    dataset = data[args.split]
    if not has_labels(dataset):
        parser.error(f"split '{args.split}' of '{args.release}' has no labels to plot.")

    distributions = sort_distributions(compute_attribute_distributions(dataset),
                                       by_attribute=args.sort_attributes,
                                       by_class=args.sort_classes)
    ordering = "" if args.sort_attributes == args.sort_classes == "schema" else (
        f", by {args.sort_attributes} attribute / {args.sort_classes} class order")
    title = (f"{args.release} / {args.split} – class frequencies"
             f" ({len(dataset):,} examples{ordering})")
    fig = plot_class_frequencies(distributions, title=title, normalize=args.normalize,
                                 log=not args.linear)

    output = args.output or Path(f"{args.release}_{args.split}_class_frequencies.png")
    fig.savefig(output, dpi=args.dpi)
    labeled = annotated(distributions)
    print(f"Wrote {output} ({sum(d.num_classes for d in labeled)} classes over"
          f" {len(labeled)} annotated attributes)")


if __name__ == "__main__":
    main()
