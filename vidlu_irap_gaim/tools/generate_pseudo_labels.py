"""Offline pseudo-label generation for semi-supervised learning.

This module generates hard pseudo-labels for unlabeled data using a frozen
pre-trained teacher model with per-attribute confidence thresholding.
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import typing as T

from vidlu_irap_gaim.training.semisup import get_hard_pseudo_labels


def generate_pseudo_labels(
    model: torch.nn.Module,
    dataset,
    conf_thresh: float = 0.0,
    temperature: float = 1.0,
    batch_size: int = 32,
    device: str = 'cuda',
) -> dict:
    """Generate hard pseudo-labels for an unlabeled dataset.

    Args:
        model: Pre-trained frozen teacher model. Should accept input and return
               a tuple of (B, K_i) logit tensors (one per attribute).
        dataset: Unlabeled dataset to generate pseudo-labels for.
        conf_thresh: Confidence threshold (float). Predictions with max probability
                     <= threshold are masked (-1).
        temperature: Temperature for softmax scaling. >1 flattens, <1 sharpens.
        batch_size: Batch size for inference.
        device: Device to run inference on ('cuda' or 'cpu').

    Returns:
        Dictionary with:
        - 'labels': (N, A) int16 array, -1 for low-confidence predictions
        - 'confidences': (N, A) float32 array, max softmax probability per (sample, attr)
        - 'segment_ids': (N,) str array (if dataset provides 'segment_id' field)
    """
    model = model.to(device)
    model.eval()

    N = len(dataset)
    # Determine number of attributes by running first sample
    with torch.no_grad():
        sample = dataset[0]
        x_sample = sample.get('rgb', sample.get('input')) if isinstance(sample, dict) else sample[0]
        if not isinstance(x_sample, torch.Tensor):
            x_sample = torch.tensor(x_sample)
        if x_sample.dim() == 3:  # (C, H, W)
            x_sample = x_sample.unsqueeze(0)  # (1, C, H, W)
        out_sample = model(x_sample.to(device))
        A = len(out_sample) if isinstance(out_sample, (tuple, list)) else 1

    # Initialize arrays
    labels = np.full((N, A), -1, dtype=np.int16)
    confidences = np.zeros((N, A), dtype=np.float32)
    segment_ids = []
    has_segment_id = False

    # Run inference on all samples
    with torch.no_grad():
        for batch_start in tqdm(range(0, N, batch_size), desc="Generating pseudo-labels"):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = list(range(batch_start, batch_end))
            batch_samples = [dataset[i] for i in batch_indices]

            # Extract input tensors
            x_batch = []
            for sample in batch_samples:
                if isinstance(sample, dict):
                    x = sample.get('rgb', sample.get('input'))
                else:
                    x = sample[0]
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                x_batch.append(x)

            # Stack and move to device
            x_batch = torch.stack(x_batch).to(device)

            # Run model
            out_batch = model(x_batch)

            # Convert to tuple if needed
            if not isinstance(out_batch, (tuple, list)):
                out_batch = (out_batch,)

            # Generate pseudo-labels
            batch_labels, batch_mask = get_hard_pseudo_labels(
                out_batch, temperature=temperature, conf_thresh=conf_thresh
            )

            # Extract confidences
            batch_confidences = []
            for logits in out_batch:
                probs = torch.softmax(logits / temperature if temperature != 1.0 else logits, dim=1)
                max_probs = probs.max(dim=1).values
                batch_confidences.append(max_probs.cpu().numpy())
            batch_confidences = np.stack(batch_confidences, axis=1)

            # Store results
            labels[batch_start:batch_end] = batch_labels.cpu().numpy().astype(np.int16)
            confidences[batch_start:batch_end] = batch_confidences.astype(np.float32)

            # Collect segment_ids if available
            for sample in batch_samples:
                if isinstance(sample, dict) and 'segment_id' in sample:
                    segment_ids.append(sample['segment_id'])
                    has_segment_id = True

    # Build result dictionary
    result = {
        'labels': labels,
        'confidences': confidences,
    }

    if has_segment_id:
        result['segment_ids'] = np.array(segment_ids, dtype=object)

    # Compute and print coverage
    valid_mask = (labels != -1)
    coverage = valid_mask.sum() / (N * A) if N > 0 else 0.0
    coverage_per_attr = valid_mask.sum(axis=0) / N if N > 0 else np.zeros(A)

    print(f"\nPseudo-label coverage:")
    print(f"  Overall: {coverage:.1%} ({valid_mask.sum()} / {N * A})")
    for i, cov in enumerate(coverage_per_attr):
        print(f"  Attribute {i}: {cov:.1%}")

    return result


def save_pseudo_labels(
    result: dict,
    output_path: T.Union[str, Path],
) -> None:
    """Save pseudo-labels to a compressed .npz file.

    Args:
        result: Dictionary from generate_pseudo_labels().
        output_path: Path to save .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(f"Saved pseudo-labels to {output_path}")


def run_pipeline(e, output: T.Union[str, Path], conf_thresh: float = 0.0, temperature: float = 1.0) -> None:
    """Run pseudo-label generation pipeline from a TrainingExperiment.

    Intended to be called as a module hook via run.py:
        test -r best -m vidlu_irap_gaim.tools.generate_pseudo_labels:run_pipeline,e,output='...',conf_thresh=0.8

    Args:
        e: TrainingExperiment with model already loaded from best checkpoint.
        output: Path to save the resulting .npz file.
        conf_thresh: Confidence threshold; predictions below are masked (-1).
        temperature: Temperature for softmax scaling.
    """
    device = next(e.trainer.model.parameters()).device
    result = generate_pseudo_labels(
        model=e.trainer.model,
        dataset=e.data.train_u,
        conf_thresh=conf_thresh,
        temperature=temperature,
        device=str(device),
    )
    save_pseudo_labels(result, output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for unlabeled data using a pre-trained teacher."
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset specification (e.g., "bih/train_unlabeled")')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for .npz file')
    parser.add_argument('--conf-thresh', type=float, default=0.0,
                       help='Confidence threshold (default: 0.0, no masking)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softmax scaling (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (default: cuda)')

    args = parser.parse_args()

    # Note: Dataset and model loading should be done by the caller
    # This script is primarily designed for programmatic use.
    print("Please use this script programmatically by importing generate_pseudo_labels().")
    print("Example:")
    print("  from vidlu_irap_gaim.tools.generate_pseudo_labels import generate_pseudo_labels, save_pseudo_labels")
    print("  result = generate_pseudo_labels(model, dataset, conf_thresh=0.8)")
    print("  save_pseudo_labels(result, 'pseudo_labels.npz')")
