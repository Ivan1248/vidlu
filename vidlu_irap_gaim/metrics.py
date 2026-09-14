"""The iRAP evaluation protocol: which metrics, over which attributes.

The machinery is `vidlu.metrics.MultiAttributeClassificationMetrics`; this module only selects
metric names and thresholds and maps iRAP attribute names to output indices and class counts.
"""

from collections.abc import Sequence

from vidlu.experiments import console_hidden
from vidlu.metrics import (
    SUPPORT_SUFFIX,
    AttributeSpec,
    ClassificationMetrics,
    MultiAttributeClassificationMetrics,
    OutputKind,
)

# iRAP evaluation metric names and defaults.
IRAP_ATTRIBUTE_METRIC_NAMES = ("mF1", "mP", "mR", "MCC")
IRAP_MAIN_METRIC = "mF1"  # "amF1" for the multi-attribute models
# Macro averages over the classes present in the split, as in Kačan et al. (2025), rather than
# counting absent classes as 0.
IRAP_IGNORE_MISSING_CLASSES = True
# Default minimum class support thresholds for restricted metrics (_suppN).
IRAP_CLASS_SUPPORT_THRESHOLDS = (5, 10)


def irap_metric_names(min_class_supports: Sequence[int] = IRAP_CLASS_SUPPORT_THRESHOLDS,
                      output_kind: OutputKind = "logits") -> dict[str, str]:
    """The metrics `get_irap_metrics` requests, as result key -> metric name, console scalars
    first.

    Scalars shown on the console: the attribute averages of the per-attribute metrics (the
    macro metrics and the chance-corrected `MCC`) and of accuracy, for probabilistic
    outputs the proper scoring rules
    `aNLL`, `aBrier` and the class-balanced `amNLL`, and per support threshold the restricted
    `amF1_suppN` (and `amNLL_suppN`) beside the unrestricted ones, so that the two can be
    compared directly rather than one replacing the other. The per-attribute metrics get
    `console_hidden` keys: kept for `MultiAttributeScorePrinter` and the tracker, off the
    console line; `nc_suppN` is the number of classes entering a restricted mean, without
    which it cannot be read.
    """
    probabilistic = output_kind != "hard"
    shown = [*("a" + name for name in IRAP_ATTRIBUTE_METRIC_NAMES), "aA"]
    hidden = ["mF1", "A", "n", "MCC"]
    if probabilistic:
        shown += ["aNLL", "aBrier", "amNLL"]
        hidden += ["NLL", "Brier", "mNLL"]
    for n in min_class_supports:
        shown.append(f"a{IRAP_MAIN_METRIC}{SUPPORT_SUFFIX}{n}")
        hidden += [f"{IRAP_MAIN_METRIC}{SUPPORT_SUFFIX}{n}", f"nc{SUPPORT_SUFFIX}{n}"]
        if probabilistic:
            shown.append(f"amNLL{SUPPORT_SUFFIX}{n}")
    return {**{name: name for name in shown}, **{console_hidden(name): name for name in hidden}}


def get_irap_metrics(
    dataset=None,
    class_counts: tuple[int, ...] | None = None,
    attrs_to_include: tuple[str, ...] | None = None,
    min_class_supports: Sequence[int] = IRAP_CLASS_SUPPORT_THRESHOLDS,
    output_kind: OutputKind = "logits",
) -> MultiAttributeClassificationMetrics:
    """Creates the iRAP evaluation metric over the canonical attribute subset.

    Args:
        dataset: Dataset whose `info.attr_to_value_to_class_idx` gives the attribute order
            (and thus the output indices) and whose `info.attr_to_num_labeled` tells which
            attributes have labels. None loads the BiH training split.
        class_counts: Number of classes per attribute in the dataset's attribute order.
            None uses `dataset.info.class_counts`.
        attrs_to_include: Attribute names to evaluate. None means the canonical subset
            (`irap_data.attrs.get_attrs_to_include`) minus the attributes with no labeled
            example in `dataset` (e.g. IRAP-Vietnam's BH-only attributes), which would
            otherwise score NaN from an empty confusion matrix.
        min_class_supports: Support thresholds for the restricted variants of the main
            metric; see `irap_metric_names`. Pass `()` for the unrestricted metrics only.
        output_kind: What the evaluation step puts in `iter_result.out`: 'logits'
            (`SupervisedStep`, the default), 'probs' (`MultiScaleSupervisedStep`) or 'hard'
            (one-hot pseudo-logits: VLM text parsing, random baselines). Metrics come from
            `--metrics`, not from the trainer, so this cannot be checked against the eval
            step; a wrong value gives wrong `NLL`/`Brier`, silently for 'logits' vs 'probs'.
            With 'hard' the probabilistic metrics are left out.
    """
    from irap_data.attrs import (
        filter_labeled_attrs,
        get_attrs_to_include,
        map_attr_names_to_indices,
    )

    if dataset is None:
        from irap_data import make_bih_data

        dataset = make_bih_data()["train"]

    if class_counts is None:
        if not hasattr(dataset, "info") or not hasattr(dataset.info, "class_counts"):
            raise ValueError("Dataset must have info.class_counts or class_counts must be provided")
        class_counts = dataset.info.class_counts

    if attrs_to_include is None:
        attrs_to_include = filter_labeled_attrs(
            get_attrs_to_include(), dataset.info.attr_to_num_labeled)

    indices = map_attr_names_to_indices(attrs_to_include,
                                        dataset.info.attr_to_value_to_class_idx.keys())
    attributes = {}
    for name, idx in zip(attrs_to_include, indices):
        if idx >= len(class_counts):
            raise ValueError(f"Attribute {name!r} has index {idx}, but class_counts has only"
                             f" {len(class_counts)} entries.")
        attributes[name] = AttributeSpec(index=idx, class_count=class_counts[idx])

    return MultiAttributeClassificationMetrics(
        attributes, metrics=irap_metric_names(min_class_supports, output_kind),
        output_kind=output_kind, ignore_missing_classes=IRAP_IGNORE_MISSING_CLASSES)


def get_irap_attribute_metrics(class_count: int) -> ClassificationMetrics:
    """Creates evaluation metrics for a single attribute in sequential enhancement.

    Computes macro F1, precision, recall, and instance count matching multi-attribute definitions.

    Args:
        class_count: Number of classes for the attribute.

    Returns:
        Configured `ClassificationMetrics` instance.
    """
    return ClassificationMetrics(class_count=class_count,
                                 metrics=(*IRAP_ATTRIBUTE_METRIC_NAMES, "n"),
                                 ignore_missing_classes=IRAP_IGNORE_MISSING_CLASSES)
