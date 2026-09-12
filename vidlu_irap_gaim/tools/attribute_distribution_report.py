"""Generate class distribution and frequency statistics for iRAP datasets.

Analyzes class representation and imbalance per attribute across splits.
"""

import argparse
import dataclasses as dc
import json
import sys
import typing as T
from pathlib import Path

import numpy as np

# `irap_data` is not installed; it lives at <repo>/irap-data (src-layout) and is normally
# put on sys.path by scripts/_context.py. Repeating that here, as `dataset_split_sizes.py`
# does, lets this run as a plain file without importing the `vidlu_irap_gaim` package, whose
# model and VLM imports this metadata-only tool does not need.
_irap_data_project = Path(__file__).resolve().parents[2] / "irap-data"
if _irap_data_project.is_dir() and str(_irap_data_project) not in sys.path:
    sys.path.insert(0, str(_irap_data_project))

from irap_data import IRAP_DATASET_FACTORIES, make_irap_data_by_name  # noqa: E402
from irap_data.irap_dataset import compute_class_occurrence_counts  # noqa: E402


@dc.dataclass(frozen=True)
class AttributeDistribution:
    """Class frequency distribution for a single attribute within a dataset split.

    Attributes:
        name: Attribute name.
        class_names: Names corresponding to class indices.
        counts: Number of occurrences per class index.
    """

    name: str
    class_names: tuple[str, ...]
    counts: np.ndarray

    @property
    def num_labeled(self) -> int:
        return int(self.counts.sum())

    @property
    def num_classes(self) -> int:
        return len(self.counts)

    @property
    def num_absent_classes(self) -> int:
        """Number of classes with zero observed examples."""
        return int((self.counts == 0).sum())

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of the most frequent to least frequent observed class count."""
        observed = self.counts[self.counts > 0]
        return float(observed.max() / observed.min()) if len(observed) else float("nan")

    @property
    def rarest_observed_share(self) -> float:
        """Fraction of total labeled examples represented by the rarest observed class."""
        observed = self.counts[self.counts > 0]
        return float(observed.min() / self.num_labeled) if len(observed) else float("nan")


def has_labels(dataset) -> bool:
    """Whether any attribute of a split is labelled, i.e. whether a class distribution is defined.

    Vietnam ships `unlabeled_train`/`val`/`test`/`unlocated` splits for semi-supervised use.
    Those keep their segments with every label set to the ignore index.
    """
    return any(n > 0 for n in dataset.info.attr_to_num_labeled.values())


def compute_attribute_distributions(dataset) -> list[AttributeDistribution]:
    """Computes the per-attribute class distribution of a split.

    Args:
        dataset: An iRAP dataset whose `info` carries `segment_ids`,
            `segment_id_to_labels`, `class_counts` and `attr_to_value_to_class_idx`.

    Returns:
        One :class:`AttributeDistribution` per attribute, in schema order.
    """
    info = dataset.info
    counts = compute_class_occurrence_counts(info)
    return [AttributeDistribution(name=attr,
                                  class_names=tuple(sorted(value_to_class_idx,
                                                           key=value_to_class_idx.__getitem__)),
                                  counts=counts[attr])
            for attr, value_to_class_idx in info.attr_to_value_to_class_idx.items()]


# Reporting #######################################################################################


def annotated(distributions: list[AttributeDistribution]) -> list[AttributeDistribution]:
    """Filters distributions to only those with labeled examples (`num_labeled > 0`)."""
    return [d for d in distributions if d.num_labeled > 0]


def unannotated_names(distributions: list[AttributeDistribution]) -> list[str]:
    return [d.name for d in distributions if d.num_labeled == 0]


def _imbalance_sort_key(d: AttributeDistribution) -> float:
    # An attribute with nothing observed has no defined ratio; put those last rather than
    # letting NaN make the ordering arbitrary.
    ratio = d.imbalance_ratio
    return -ratio if np.isfinite(ratio) else float("inf")


_ATTRIBUTE_SORT_KEYS = {
    "count": lambda d: -d.num_labeled,
    "imbalance": _imbalance_sort_key,
}

#: How to order attribute groups. `schema` is the canonical attribute order of the
#: release; the others put the attributes a reader is looking for at the top. `count` is
#: the same key as in :data:`CLASS_ORDERS` – the number of labelled examples – applied one
#: level up, so the two orderings share a name rather than inventing one each.
ATTRIBUTE_ORDERS = ("schema", *_ATTRIBUTE_SORT_KEYS)

#: How to order classes within an attribute. `schema` is the class-index order, which is
#: often meaningful (ordinal attributes such as speed limits); `count` ranks by frequency.
CLASS_ORDERS = ("schema", "count")


def sort_distributions(distributions: list[AttributeDistribution], *,
                       by_attribute: str = "schema",
                       by_class: str = "schema") -> list[AttributeDistribution]:
    """Reorder attribute groups and the classes within them.

    Args:
        by_attribute: One of :data:`ATTRIBUTE_ORDERS`. `count` sorts by descending number
            of labelled examples, `imbalance` by descending max:min class ratio, which
            brings the most skewed attributes to the top.
        by_class: One of :data:`CLASS_ORDERS`. `count` sorts classes within each attribute
            by descending frequency.

    Sorting is stable, so ties keep the schema order.
    """
    if by_class not in CLASS_ORDERS:
        raise ValueError(f"{by_class=} is not one of {CLASS_ORDERS}.")
    if by_class == "count":
        distributions = [_with_classes_sorted_by_count(d) for d in distributions]
    if by_attribute != "schema":
        distributions = sorted(distributions, key=_ATTRIBUTE_SORT_KEYS[by_attribute])
    return distributions


def _with_classes_sorted_by_count(d: AttributeDistribution) -> AttributeDistribution:
    order = np.argsort(-d.counts, kind="stable")
    return dc.replace(d, class_names=tuple(d.class_names[i] for i in order),
                      counts=d.counts[order])


def rare_classes(distributions: list[AttributeDistribution], *,
                 rare_fraction: float) -> list[tuple[int, float, str, str]]:
    """Classes whose share of their attribute's labelled examples is below `rare_fraction`.

    Returns:
        `(count, share, attribute, class_name)` tuples, rarest first.
    """
    rare = [(int(count), float(count / d.num_labeled), d.name, class_name)
            for d in annotated(distributions)
            for class_name, count in zip(d.class_names, d.counts)
            if count / d.num_labeled < rare_fraction]
    return sorted(rare, key=lambda r: (r[0], r[2]))


_ATTRIBUTE_COLUMN_WIDTH = 44


def format_attribute_table(distributions: list[AttributeDistribution]) -> str:
    """One row per annotated attribute: how much is labelled and how skewed it is."""
    header = (f"{'attribute':<{_ATTRIBUTE_COLUMN_WIDTH}}{'labeled':>9}{'classes':>9}"
              f"{'absent':>8}{'max:min':>10}{'rarest':>9}")
    rows = [header, "-" * len(header)]
    for d in annotated(distributions):
        rows.append(f"{d.name:<{_ATTRIBUTE_COLUMN_WIDTH}}{d.num_labeled:>9}{d.num_classes:>9}"
                    f"{d.num_absent_classes:>8}{d.imbalance_ratio:>10,.0f}"
                    f"{d.rarest_observed_share:>8.2%}")
    return "\n".join(rows)


def format_rare_classes(distributions: list[AttributeDistribution], *,
                        rare_fraction: float, max_rows: int | None = None) -> str:
    """Formats the table of :func:`rare_classes`, truncated to `max_rows` if given."""
    rare = rare_classes(distributions, rare_fraction=rare_fraction)
    header = f"{'count':>8}{'share':>9}  {'attribute':<{_ATTRIBUTE_COLUMN_WIDTH}}class"
    rows = [header, "-" * (len(header) + 16)]  # room for a class name under `class`
    shown = rare if max_rows is None else rare[:max_rows]
    for count, share, attr, class_name in shown:
        rows.append(f"{count:>8,}{share:>8.3%}  {attr:<{_ATTRIBUTE_COLUMN_WIDTH}}{class_name}")
    if max_rows is not None and len(rare) > max_rows:
        rows.append(f"... and {len(rare) - max_rows} more below {rare_fraction:.1%}")
    return "\n".join(rows)


def summarize(distributions: list[AttributeDistribution], *, rare_fraction: float) -> dict:
    """Counts over the annotated attributes; the unannotated ones are named separately."""
    labeled = annotated(distributions)
    return dict(
        num_attributes=len(distributions),
        num_annotated_attributes=len(labeled),
        unannotated_attributes=unannotated_names(distributions),
        num_classes=sum(d.num_classes for d in labeled),
        num_absent_classes=sum(d.num_absent_classes for d in labeled),
        num_rare_classes=len(rare_classes(labeled, rare_fraction=rare_fraction)),
        rare_fraction=rare_fraction,
    )


def to_json_record(distributions: list[AttributeDistribution]) -> list[dict]:
    """Converts attribute distributions into JSON-serializable dictionaries."""
    return [dict(attribute=d.name, num_labeled=d.num_labeled,
                 num_absent_classes=d.num_absent_classes,
                 imbalance_ratio=None if d.num_labeled == 0 else d.imbalance_ratio,
                 classes=[dict(name=n, count=int(c),
                               share=None if d.num_labeled == 0 else float(c / d.num_labeled))
                          for n, c in zip(d.class_names, d.counts)])
            for d in distributions]


def main(argv: T.Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report per-attribute class distributions for iRAP releases.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--releases", nargs="+", default=["bih", "vietnam"],
                        choices=list(IRAP_DATASET_FACTORIES),
                        help="Releases to report (default: bih vietnam).")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to report. Missing ones (e.g. Vietnam ships no"
                             " non-empty test split) and unlabelled ones are skipped with a"
                             " note (default: train val test).")
    parser.add_argument("--rare-fraction", type=float, default=0.01,
                        help="A class is 'rare' below this share of its attribute's"
                             " labelled examples (default: 0.01).")
    parser.add_argument("--max-rare-rows", type=int, default=40,
                        help="Truncate the rare-class listing (default: 40, 0 for all).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Also write the full distributions to this JSON file.")
    args = parser.parse_args(argv)

    max_rare_rows = None if args.max_rare_rows == 0 else args.max_rare_rows
    report = {}
    for release in args.releases:
        data = make_irap_data_by_name(release)
        for split in args.splits:
            if split not in data:
                print(f"\n[{release}] split '{split}' not present; skipping.")
                continue
            dataset = data[split]
            if not has_labels(dataset):
                print(f"\n[{release}] split '{split}' has no labels; skipping.")
                continue
            distributions = compute_attribute_distributions(dataset)
            summary = summarize(distributions, rare_fraction=args.rare_fraction)

            title = f"{release} / {split} – {len(dataset):,} examples"
            print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
            print(f"{summary['num_annotated_attributes']}/{summary['num_attributes']}"
                  f" attributes annotated, covering {summary['num_classes']} classes,"
                  f" of which {summary['num_absent_classes']} are never observed and"
                  f" {summary['num_rare_classes']} fall below {args.rare_fraction:.1%}.")
            # Named once here, then excluded from the table and the rare-class listing.
            if unannotated := summary["unannotated_attributes"]:
                print(f"\nNot annotated by this release ({len(unannotated)}), excluded below:")
                for name in unannotated:
                    print(f"  - {name}")
            print()
            print(format_attribute_table(distributions))
            print(f"\nClasses below {args.rare_fraction:.1%} of their attribute:\n")
            print(format_rare_classes(distributions, rare_fraction=args.rare_fraction,
                                      max_rows=max_rare_rows))

            report[f"{release}/{split}"] = dict(
                num_examples=len(dataset), summary=summary,
                attributes=to_json_record(distributions))

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
