"""
Compare evaluation runs that differ in how many attributes share a VLM session.

Reads the ``records.jsonl`` files ``vlm_inference.run_evaluation`` writes -- one
record per (segment, attribute) -- and reports what changed between arms.

Every arm covers the same segments, so the comparison is *paired*: correctness is
compared segment by segment rather than through two independent aggregates. That
is what makes a small difference in ``amF1`` interpretable, and it is why the
per-(segment, attribute) records exist at all.

Usage:
    python -m vidlu_irap_gaim.tools.compare_session_modes \\
        zs_all=results/zs_all/test/records.jsonl \\
        zs_per=results/zs_per/test/records.jsonl \\
        ft_all=results/ft_all/test/records.jsonl \\
        ft_per=results/ft_per/test/records.jsonl \\
        --baseline zs_all --output report.md

Arms may be given as ``name=path`` or as bare paths (named after their parent
directory). The first arm is the baseline unless ``--baseline`` says otherwise.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

# Number of resamples for the bootstrap CI on the paired metric difference.
DEFAULT_NUM_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
# Two-sided significance level, before multiplicity correction.
ALPHA = 0.05


@dataclass
class Arm:
    """One evaluation run's records, indexed for paired comparison."""

    name: str
    path: Path
    records: list[dict] = field(default_factory=list)

    # `records` is loaded once and never appended to, so these indices are built on
    # first use instead of on every access -- `segment_ids` alone is read once per arm
    # pair and once per arm inside the report's loops.

    @cached_property
    def segment_ids(self) -> set[str]:
        return {r["segment_id"] for r in self.records}

    @cached_property
    def attrs(self) -> list[str]:
        seen = {}
        for r in self.records:
            seen.setdefault(r["attr"], None)
        return list(seen)

    @cached_property
    def by_key(self) -> dict[tuple[str, str], dict]:
        return {(r["segment_id"], r["attr"]): r for r in self.records}


def load_arm(spec: str) -> Arm:
    """Loads one ``name=path`` (or bare path) argument."""
    name, _, path_str = spec.partition("=")
    if not path_str:
        path_str, name = name, Path(name).parent.name or Path(name).stem
    path = Path(path_str)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} holds no records.")
    return Arm(name=name, path=path, records=records)


def is_scored(record: dict) -> bool:
    """Whether the record has a label to be judged against.

    Unlabeled attributes carry ``IGNORE_LABEL_INDEX`` and are dropped by the
    metrics too, so they must not enter the paired counts either.
    """
    return record["target_idx"] is not None and record["target_idx"] >= 0


def is_correct(record: dict, invalid_policy: str = "class0") -> bool | None:
    """Whether the record's prediction matches the label.

    ``invalid_policy`` mirrors the scoring in
    ``vidlu_irap_gaim.vlm.predictions``: "class0" judges an unusable response as a
    prediction of class 0, "ignore" returns None so the pair is dropped.
    """
    if not is_scored(record):
        return None
    if record["valid"]:
        return record["pred_idx"] == record["target_idx"]
    if invalid_policy == "ignore":
        return None
    return record["target_idx"] == 0


# ----- metrics ----------------------------------------------------------------

def macro_f1(records, invalid_policy: str = "class0") -> float:
    """Macro-F1 over the classes present in ``records`` for one attribute.

    Matches `vidlu.metrics.ClassificationMetrics` with
    ``ignore_missing_classes``: a class absent from both truth and prediction
    contributes nothing rather than a zero.
    """
    tp, predicted, actual = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in records:
        if not is_scored(r):
            continue
        if not r["valid"]:
            if invalid_policy == "ignore":
                continue
            pred = 0
        else:
            pred = r["pred_idx"]
        target = r["target_idx"]
        predicted[pred] += 1
        actual[target] += 1
        if pred == target:
            tp[target] += 1

    f1s = []
    for class_idx in set(predicted) | set(actual):
        denominator = predicted[class_idx] + actual[class_idx]
        if denominator == 0:
            continue
        f1s.append(2 * tp[class_idx] / denominator)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def arm_summary(arm: Arm, invalid_policy: str = "class0") -> dict:
    """Aggregate numbers for one arm."""
    scored = [r for r in arm.records if is_scored(r)]
    correct = [is_correct(r, invalid_policy) for r in scored]
    judged = [c for c in correct if c is not None]
    per_attr = defaultdict(list)
    for r in arm.records:
        per_attr[r["attr"]].append(r)
    attr_f1 = {a: macro_f1(rs, invalid_policy) for a, rs in per_attr.items()}
    finite_f1 = [v for v in attr_f1.values() if not math.isnan(v)]

    sessions = {(r["segment_id"], r["session_idx"]) for r in arm.records}
    truncated = {(r["segment_id"], r["session_idx"]) for r in arm.records
                 if r["is_truncated"]}
    multiline = {(r["segment_id"], r["session_idx"]) for r in arm.records
                 if r["num_attrs_in_session"] == 1 and (r["num_response_lines"] or 0) > 1}
    response_tokens = [r["num_response_tokens"] for r in arm.records
                       if r["num_response_tokens"] is not None]

    return {
        "num_segments": len(arm.segment_ids),
        "num_attrs": len(attr_f1),
        "num_scored": len(scored),
        "amF1": (sum(finite_f1) / len(finite_f1)) if finite_f1 else float("nan"),
        "acc": (sum(judged) / len(judged)) if judged else float("nan"),
        "invalid_rate": (sum(1 for r in scored if not r["valid"]) / len(scored)
                         if scored else float("nan")),
        "num_sessions": len(sessions),
        "truncation_rate": len(truncated) / len(sessions) if sessions else float("nan"),
        # Only meaningful for single-attribute sessions: a model asked about one
        # attribute that responses several lines is responding the *other* format.
        "multiline_rate": (len(multiline) / len(sessions)
                           if sessions and any(r["num_attrs_in_session"] == 1
                                               for r in arm.records) else float("nan")),
        "mean_response_tokens": (sum(response_tokens) / len(response_tokens)
                                 if response_tokens else float("nan")),
        "attr_f1": attr_f1,
    }


# ----- paired statistics ------------------------------------------------------

def mcnemar_per_attribute(
    baseline: Arm,
    other: Arm,
    invalid_policy: str = "class0",
) -> dict[str, tuple[int, int, float]]:
    """Exact McNemar test per attribute, on per-segment correctness.

    The arms score the same segments, so the informative quantity is the
    discordant pairs: how often one arm is right where the other is wrong.
    Every attribute is counted in one pass over the records, since indexing the
    records once per attribute would be quadratic in the attribute count.

    Returns:
        ``{attr: (num_baseline_only, num_other_only, p_value)}``.
    """
    other_by_key = other.by_key
    discordant = defaultdict(lambda: [0, 0])
    for key, base_record in baseline.by_key.items():
        other_record = other_by_key.get(key)
        if other_record is None:
            continue
        base_correct = is_correct(base_record, invalid_policy)
        other_correct = is_correct(other_record, invalid_policy)
        if base_correct is None or other_correct is None:
            continue
        if base_correct and not other_correct:
            discordant[key[1]][0] += 1
        elif other_correct and not base_correct:
            discordant[key[1]][1] += 1

    return {attr: (base_only, other_only,
                   _binomial_two_sided_p(base_only, other_only))
            for attr, (base_only, other_only) in discordant.items()}


def _binomial_two_sided_p(a: int, b: int) -> float:
    """Two-sided exact binomial p at p=0.5 over the a+b discordant pairs."""
    n = a + b
    if n == 0:
        return float("nan")
    smaller = min(a, b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def benjamini_hochberg(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values.

    One test per attribute means dozens of them; without correction, several would be
    expected to look significant at 0.05 by chance alone.
    """
    finite = {k: p for k, p in p_values.items() if not math.isnan(p)}
    ordered = sorted(finite.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running_min = {}, 1.0
    for rank, (key, p) in enumerate(reversed(ordered), start=1):
        index = n - rank + 1
        running_min = min(running_min, p * n / index)
        adjusted[key] = running_min
    for key in p_values:
        adjusted.setdefault(key, float("nan"))
    return adjusted


def bootstrap_amf1_difference(
    baseline: Arm,
    other: Arm,
    invalid_policy: str = "class0",
    num_resamples: int = DEFAULT_NUM_BOOTSTRAP,
) -> tuple[float, float, float]:
    """Percentile CI for ``amF1(other) - amF1(baseline)``, resampling segments.

    Segments are the independent unit -- one segment's attributes are all read off
    the same image and are not independent of each other -- so the resample is over
    segments, carrying all of a segment's records with it.
    """
    shared = sorted(baseline.segment_ids & other.segment_ids)
    base_by_segment, other_by_segment = defaultdict(list), defaultdict(list)
    for r in baseline.records:
        base_by_segment[r["segment_id"]].append(r)
    for r in other.records:
        other_by_segment[r["segment_id"]].append(r)

    def amf1(records) -> float:
        per_attr = defaultdict(list)
        for r in records:
            per_attr[r["attr"]].append(r)
        values = [macro_f1(rs, invalid_policy) for rs in per_attr.values()]
        finite = [v for v in values if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    observed = amf1([r for s in shared for r in other_by_segment[s]]) - \
        amf1([r for s in shared for r in base_by_segment[s]])

    rng = random.Random(BOOTSTRAP_SEED)
    differences = []
    for _ in range(num_resamples):
        sample = [shared[rng.randrange(len(shared))] for _ in shared]
        differences.append(
            amf1([r for s in sample for r in other_by_segment[s]])
            - amf1([r for s in sample for r in base_by_segment[s]]))
    differences.sort()
    low = differences[int(0.025 * len(differences))]
    high = differences[min(len(differences) - 1, int(0.975 * len(differences)))]
    return observed, low, high


def position_effect(arm: Arm, invalid_policy: str = "class0") -> tuple[float, float]:
    """Slope of correctness on the attribute's position in the prompt.

    Only meaningful where a session lists several attributes: it tests whether
    attributes late in a long prompt are responded worse, which is the mechanism
    per-attribute sessions are supposed to remove. Returns ``(slope, intercept)``
    from an ordinary least-squares fit, or NaNs when the arm has no multi-
    attribute sessions.
    """
    points = [(r["attr_position"], is_correct(r, invalid_policy))
              for r in arm.records
              if r["num_attrs_in_session"] > 1 and r["attr_position"] is not None]
    points = [(x, float(y)) for x, y in points if y is not None]
    if len(points) < 2:
        return float("nan"), float("nan")
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    if variance == 0:
        return float("nan"), float("nan")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return slope, mean_y - slope * mean_x


def marginal_shift(arm: Arm, invalid_policy: str = "class0") -> dict[str, float]:
    """Per attribute, the total-variation distance between the predicted class
    distribution and the label distribution.

    A model fine-tuned on one prompt learns that prompt's label priors along with
    the visual task. Accuracy can therefore fall for two very different reasons:
    the responses stay calibrated to the attribute's marginal and are merely wrong
    on individual images, or the marginal itself moves. This separates the two.
    0 means the predicted and observed class frequencies coincide (which says
    nothing about per-image correctness); 1 means disjoint support.
    """
    predicted: dict[str, Counter] = defaultdict(Counter)
    actual: dict[str, Counter] = defaultdict(Counter)
    for r in arm.records:
        if not is_scored(r):
            continue
        if not r["valid"]:
            if invalid_policy == "ignore":
                continue
            pred = 0
        else:
            pred = r["pred_idx"]
        predicted[r["attr"]][pred] += 1
        actual[r["attr"]][r["target_idx"]] += 1

    result = {}
    for attr, actual_counts in actual.items():
        predicted_counts = predicted[attr]
        num_predicted = sum(predicted_counts.values())
        num_actual = sum(actual_counts.values())
        if num_predicted == 0 or num_actual == 0:
            result[attr] = float("nan")
            continue
        result[attr] = 0.5 * sum(
            abs(predicted_counts[c] / num_predicted - actual_counts[c] / num_actual)
            for c in set(predicted_counts) | set(actual_counts))
    return result


def mean_marginal_shift(arm: Arm, invalid_policy: str = "class0") -> float:
    values = [v for v in marginal_shift(arm, invalid_policy).values() if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


# ----- report -----------------------------------------------------------------

def _fmt(value, digits=4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def build_report(arms: list[Arm], baseline_name: str, num_bootstrap: int) -> str:
    """The markdown comparison report."""
    by_name = {arm.name: arm for arm in arms}
    baseline = by_name[baseline_name]
    others = [a for a in arms if a.name != baseline_name]

    lines = ["# Session-granularity comparison", ""]

    shared_segments = set.intersection(*(a.segment_ids for a in arms))
    lines.append(f"Arms: {', '.join(a.name for a in arms)}. Baseline: `{baseline_name}`.")
    lines.append(f"Segments shared by every arm: **{len(shared_segments)}**.")
    for arm in arms:
        extra = len(arm.segment_ids - shared_segments)
        if extra:
            lines.append(f"- `{arm.name}` covers {extra} segment(s) no other arm does; "
                         f"the paired statistics use the shared set only.")
    lines.append("")

    summaries = {arm.name: arm_summary(arm) for arm in arms}
    summaries_ignoring_invalid = {arm.name: arm_summary(arm, "ignore") for arm in arms}

    lines += ["## Aggregate", "",
              "`amF1`/`acc` score an unusable response as class 0; the `_ex` columns "
              "exclude it instead. Class 0 is often the majority value, so the first "
              "is generous to a model that fails to response and the second measures "
              "skill conditional on responding in format.", "",
              "| arm | amF1 | acc | amF1_ex | acc_ex | invalid | truncated | multiline "
              "| resp. tokens | sessions |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        s, sx = summaries[arm.name], summaries_ignoring_invalid[arm.name]
        lines.append(
            f"| `{arm.name}` | {_fmt(s['amF1'])} | {_fmt(s['acc'])} | {_fmt(sx['amF1'])} "
            f"| {_fmt(sx['acc'])} | {_fmt(s['invalid_rate'])} | {_fmt(s['truncation_rate'])} "
            f"| {_fmt(s['multiline_rate'])} | {_fmt(s['mean_response_tokens'], 1)} "
            f"| {s['num_sessions']} |")
    lines.append("")

    finite_truncation = [s["truncation_rate"] for s in summaries.values()
                         if not math.isnan(s["truncation_rate"])]
    worst_truncation = max(finite_truncation, default=0.0)
    if worst_truncation > 0.001:
        lines += [f"> **The comparison is not yet valid.** Truncation reaches "
                  f"{worst_truncation:.3f} in some arm. A budget that binds in one arm "
                  f"only manufactures a difference between the arms; raise "
                  f"`--response-token-margin` and rerun that arm.", ""]

    lines += ["## Paired difference in amF1", "",
              "Resampling segments, not attributes: one segment's responses all come from "
              "one image and are not independent.", "",
              "| arm | ΔamF1 vs baseline | 95% CI |", "|---|---|---|"]
    for arm in others:
        observed, low, high = bootstrap_amf1_difference(
            baseline, arm, num_resamples=num_bootstrap)
        crosses_zero = low <= 0 <= high
        lines.append(f"| `{arm.name}` | {observed:+.4f} | [{low:+.4f}, {high:+.4f}]"
                     f"{'' if crosses_zero else ' *'} |")
    lines += ["", "`*` marks a CI that excludes zero.", ""]

    lines += ["## Position effect in multi-attribute sessions", "",
              "Slope of per-response correctness on the attribute's 1-based position in "
              "the prompt. A negative slope is the mechanism per-attribute sessions "
              "remove; a slope near zero means that mechanism is not operating.", "",
              "| arm | slope per position | intercept |", "|---|---|---|"]
    for arm in arms:
        slope, intercept = position_effect(arm)
        lines.append(f"| `{arm.name}` | {_fmt(slope, 6)} | {_fmt(intercept)} |")
    lines.append("")

    lines += ["## Calibration of the predicted class marginals", "",
              "Total-variation distance between an arm's predicted class frequencies "
              "and the label frequencies, averaged over attributes. It separates two "
              "causes of a lower score: responses that stay on the attribute's marginal "
              "but land on the wrong images, versus a marginal that has itself moved -- "
              "the signature of a model that has lost the priors it was fine-tuned on. "
              "A low distance is not evidence of accuracy.", "",
              "| arm | mean TVD | worst attributes |", "|---|---|---|"]
    for arm in arms:
        shifts = marginal_shift(arm)
        worst = sorted((a for a in shifts if not math.isnan(shifts[a])),
                       key=lambda a: -shifts[a])[:3]
        lines.append(f"| `{arm.name}` | {_fmt(mean_marginal_shift(arm))} | "
                     + ", ".join(f"{a} ({shifts[a]:.2f})" for a in worst) + " |")
    lines.append("")

    for arm in others:
        lines += [f"## Per attribute: `{arm.name}` vs `{baseline_name}`", "",
                  "`b-only`/`o-only` are the discordant segments -- where exactly one "
                  "arm is right. `p_adj` is Benjamini-Hochberg over the attributes.", "",
                  "| attribute | baseline mF1 | arm mF1 | Δ | b-only | o-only | p | p_adj |",
                  "|---|---|---|---|---|---|---|---|"]
        base_f1 = summaries[baseline_name]["attr_f1"]
        arm_f1 = summaries[arm.name]["attr_f1"]
        stats = mcnemar_per_attribute(baseline, arm)
        adjusted = benjamini_hochberg({a: p for a, (_, _, p) in stats.items()})

        def delta_for(attr, base_f1=base_f1, arm_f1=arm_f1) -> float:
            base_value = base_f1.get(attr, float("nan"))
            arm_value = arm_f1.get(attr, float("nan"))
            if math.isnan(base_value) or math.isnan(arm_value):
                return float("nan")
            return arm_value - base_value

        # Worst regressions first: those are what a degraded arm has to explain.
        rows = sorted(arm.attrs,
                      key=lambda a: (0.0 if math.isnan(delta_for(a)) else delta_for(a)))
        for attr in rows:
            base_only, other_only, p = stats.get(attr, (0, 0, float("nan")))
            lines.append(
                f"| {attr} | {_fmt(base_f1.get(attr))} | {_fmt(arm_f1.get(attr))} "
                f"| {_fmt(delta_for(attr))} | {base_only} | {other_only} | {_fmt(p)} "
                f"| {_fmt(adjusted.get(attr, float('nan')))} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare VLM evaluation runs that differ in attributes per session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("arms", nargs="+",
                        help="records.jsonl paths, optionally as name=path")
    parser.add_argument("--baseline", default=None,
                        help="Arm to compare the others against (default: the first)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the report here as well as to stdout")
    parser.add_argument("--num-bootstrap", type=int, default=DEFAULT_NUM_BOOTSTRAP,
                        help=f"Bootstrap resamples (default: {DEFAULT_NUM_BOOTSTRAP})")
    args = parser.parse_args()

    arms = [load_arm(spec) for spec in args.arms]
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"Arm names must be distinct; got {names}. Use name=path.")
    baseline_name = args.baseline or names[0]
    if baseline_name not in names:
        raise SystemExit(f"Baseline {baseline_name!r} is not one of {names}.")

    report = build_report(arms, baseline_name, args.num_bootstrap)
    # The report is UTF-8 (it uses Δ); a Windows console defaults to cp1252 and
    # would raise rather than print it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWritten to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
