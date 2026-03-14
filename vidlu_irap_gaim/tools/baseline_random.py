"""
Baseline evaluation script for road attribute classification.

Supports three baseline prediction strategies:
- uniform: Each class equally likely (1/K probability) - theoretical chance level
- stratified: Sample according to training set class distribution
- most_common: Always predict the most frequent class from training set

Usage:
    # Evaluate all baselines
    IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.baseline_random --split test --mode all

    # Evaluate specific baseline
    IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.baseline_random --split test --mode uniform
    IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.baseline_random --split test --mode stratified
    IRAP_HOME=~/projects/irap_home python -m vidlu_irap_gaim.tools.baseline_random --split test --mode most_common
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from vidlu.utils.collections import NameDict
from vidlu_irap_gaim.data.attrs import get_attrs_to_include
from vidlu_irap_gaim.data.attribute_frequencies import AttributeFrequencyStats, compute_attribute_frequency_stats
from vidlu_irap_gaim.data import make_bih_data
from vidlu_irap_gaim.metrics import get_irap_metrics


BaselineMode = Literal["uniform", "stratified", "most_common"]
ALL_MODES: tuple[BaselineMode, ...] = ("uniform", "stratified", "most_common")


def generate_uniform_predictions(
    batch_size: int,
    class_counts: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """Generate uniform random predictions (each class equally likely)."""
    return tuple(
        torch.rand(batch_size, k)
        for k in class_counts
    )


def generate_stratified_predictions(
    batch_size: int,
    class_priors: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Generate stratified random predictions (sample from class priors)."""
    outs = []
    for prior in class_priors:
        # Sample class indices according to prior distribution
        sampled_indices = torch.multinomial(prior, batch_size, replacement=True)
        # Convert to one-hot-like probabilities
        k = len(prior)
        out = torch.zeros(batch_size, k)
        out.scatter_(1, sampled_indices.unsqueeze(1), 1.0)
        outs.append(out)
    return tuple(outs)


def generate_most_common_predictions(
    batch_size: int,
    class_counts: tuple[int, ...],
    most_common_class_indices: list[int],
) -> tuple[torch.Tensor, ...]:
    """Generate most-common-class predictions (always predict most frequent)."""
    outs = []
    for attr_idx, common_idx in enumerate(most_common_class_indices):
        k = class_counts[attr_idx]
        out = torch.zeros(batch_size, k)
        out[:, common_idx] = 1.0
        outs.append(out)
    return tuple(outs)


def generate_predictions(
    mode: BaselineMode,
    batch_size: int,
    stats: AttributeFrequencyStats,
) -> tuple[torch.Tensor, ...]:
    """Generate baseline predictions based on mode."""
    if mode == "uniform":
        return generate_uniform_predictions(batch_size, stats.class_counts)
    elif mode == "stratified":
        return generate_stratified_predictions(batch_size, stats.class_priors)
    elif mode == "most_common":
        return generate_most_common_predictions(
            batch_size, stats.class_counts, stats.most_common_class_indices
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_baseline(
    mode: BaselineMode,
    eval_ds,
    stats: AttributeFrequencyStats,
    batch_size: int = 128,
) -> dict[str, float | dict]:
    """Evaluate a single baseline mode on the evaluation dataset."""
    attrs_to_include = get_attrs_to_include()
    metrics = get_irap_metrics(eval_ds, attrs_to_include=attrs_to_include)
    for m in metrics:
        m.reset()

    num_samples = len(eval_ds)

    for i in tqdm(range(0, num_samples, batch_size), desc=f"Evaluating {mode}"):
        current_batch_size = min(batch_size, num_samples - i)

        # Collect targets for this batch
        targets = []
        for j in range(i, i + current_batch_size):
            sample = eval_ds[j]
            targets.append(sample["target"])

        targets = torch.stack(targets)  # (B, A)
        outs = generate_predictions(mode, current_batch_size, stats)

        iter_result = NameDict(out=outs, target=targets)
        for m in metrics:
            m.update(iter_result)

    # Compute final metrics
    computed_metrics = {}
    for m in metrics:
        computed_metrics.update(m.compute())

    return computed_metrics


def print_metrics(mode: str, metrics: dict[str, float | dict]) -> None:
    """Print metrics in a formatted way."""
    print(f"\n=== {mode.upper()} Baseline Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            avg = sum(v.values()) / len(v) if v else 0
            print(f"  {k} (avg): {avg:.4f}")


def run_baseline_evaluation(
    split: str = "test",
    mode: str = "all",
    output_dir: str | Path = "baseline_results",
    batch_size: int = 128,
) -> dict[str, dict]:
    """Run baseline evaluation for specified mode(s).

    Args:
        split: Dataset split to evaluate on ("train", "val", "test").
        mode: Baseline mode ("uniform", "stratified", "most_common", or "all").
        output_dir: Directory to save results.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary mapping mode names to their computed metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = make_bih_data()
    train_ds = datasets["train"]
    eval_ds = datasets[split]

    print(f"Training samples: {len(train_ds)}")
    print(f"Evaluation samples ({split}): {len(eval_ds)}")

    # Compute training statistics
    print("Computing training statistics...")
    stats = compute_attribute_frequency_stats(train_ds)
    print(f"Number of attributes: {len(stats.class_counts)}")

    # Determine which modes to evaluate
    modes_to_eval: list[BaselineMode] = list(ALL_MODES) if mode == "all" else [mode]

    # Evaluate each mode
    results = {}
    for m in modes_to_eval:
        metrics = evaluate_baseline(m, eval_ds, stats, batch_size=batch_size)
        results[m] = metrics
        print_metrics(m, metrics)

    # Save summary
    # Convert tensors to lists for JSON serialization
    class_priors_serializable = [p.tolist() for p in stats.class_priors]

    summary = {
        "split": split,
        "num_samples": len(eval_ds),
        "num_attributes": len(stats.class_counts),
        "class_counts": list(stats.class_counts),
        "most_common_classes": stats.most_common_class_indices,
        "class_priors": class_priors_serializable,
        "results": {
            mode_name: {
                k: (v if isinstance(v, (int, float)) else dict(v))
                for k, v in mode_metrics.items()
            }
            for mode_name, mode_metrics in results.items()
        },
    }

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {summary_path}")

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline classifiers for road attribute classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--mode",
        choices=["uniform", "stratified", "most_common", "all"],
        default="all",
        help="Baseline mode to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_results",
        help="Directory to save results (default: baseline_results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )

    args = parser.parse_args()

    run_baseline_evaluation(
        split=args.split,
        mode=args.mode,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
