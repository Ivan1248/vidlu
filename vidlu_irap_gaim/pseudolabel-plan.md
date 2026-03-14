# Plan: Semi-Supervised Pseudo-Label Self-Training for iRAP

## Context

The iRAP extension already has soft consistency regularization (`irap_semisup_trainer`: KL divergence between teacher and student outputs on clean vs. jittered input, same model for both). The goal here is to add **hard pseudo-label self-training**: a frozen pre-trained teacher generates hard argmax labels for unlabeled data, with per-(sample, attribute) confidence thresholding and temperature scaling. Two modes are implemented — on-the-fly (teacher runs each batch) and offline (pseudo-labels pre-generated to a .npz).

This is FixMatch-style: teacher sees **clean** `x_u`, student is trained to predict those pseudo-labels on **jittered** `x_u`.

---

## Files to Modify

- `vidlu_irap_gaim/losses.py` — add `ignore_index` support
- `vidlu_irap_gaim/semisup.py` — add `get_hard_pseudo_labels`, `PseudoLabeledDataset`
- `vidlu_irap_gaim/training.py` — add `MultiAttributePseudoLabelStep`, two trainer configs
- `vidlu_irap_gaim/tools/generate_pseudo_labels.py` — new file

---

## Step 1: Add `ignore_index` to `MultiAttributeCrossEntropyLoss` (`losses.py`)

Add `ignore_index: int = -100` parameter (default preserves backward compat with PyTorch's default).

```python
def __init__(self, reduction="mean", attrs_idx=None, ignore_index: int = -100):
    ...
    self.ignore_index = ignore_index
```

Pass to each `F.cross_entropy` call:
```python
F.cross_entropy(..., ignore_index=self.ignore_index)
```

Handle NaN from all-masked attribute (only for `reduction="mean"`): after building `per_attr_losses`, filter nans before stacking:
```python
if effective_reduction == "mean":
    valid = [l for l in per_attr_losses if not l.isnan()]
    if not valid:
        return outputs[0].new_zeros(()).requires_grad_()
    return torch.stack(valid).mean()
```

---

## Step 2: Add `get_hard_pseudo_labels` to `semisup.py`

Core shared function used by both the on-the-fly step and the generation tool. Supports both simple fixed thresholding and per-attribute adaptive thresholding (inspired by MC-PanDA++).

```python
def get_hard_pseudo_labels(
    logits_tuple: tuple,
    temperature: float = 1.0,
    conf_thresh: float | dict = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert multi-attribute logit tuple to hard pseudo-labels with confidence masking.

    Args:
        logits_tuple: Tuple of (B, K_i) logit tensors, length A (num attributes).
        temperature: Temperature for softmax scaling. >1 flattens, <1 sharpens confidence.
        conf_thresh: Confidence threshold(s). Can be:
            - float: Single global threshold applied to all attributes (fixed mode)
            - dict: Per-attribute thresholds, mapping attr_idx -> float (adaptive mode)

    Returns:
        labels: (B, A) int64 tensor, -1 where max_prob <= threshold for that attribute
        mask:   (B, A) bool tensor, True where label is valid
    """
    B = logits_tuple[0].shape[0]
    device = logits_tuple[0].device
    A = len(logits_tuple)
    labels = torch.full((B, A), -1, dtype=torch.long, device=device)
    mask = torch.zeros((B, A), dtype=torch.bool, device=device)

    for i, logits in enumerate(logits_tuple):
        scaled = logits / temperature if temperature != 1.0 else logits
        probs = torch.softmax(scaled, dim=1)
        max_prob, argmax = probs.max(dim=1)

        # Get threshold for this attribute (adaptive or fixed)
        thresh = conf_thresh[i] if isinstance(conf_thresh, dict) else conf_thresh

        valid = max_prob > thresh
        labels[:, i] = torch.where(valid, argmax, torch.full_like(argmax, -1))
        mask[:, i] = valid

    return labels, mask


def update_adaptive_thresholds(
    logits_tuple: tuple,
    thresholds: dict,
    ema_momentum: float = 0.999,
    aggregation_fn=None,
) -> dict:
    """
    Update per-attribute confidence thresholds using exponential moving average (MC-PanDA++ style).

    This allows threshold τ_a to adapt to the evolving confidence distribution of each attribute,
    automatically handling class imbalance within attributes. Each attribute may have different
    dominant/rare class distributions.

    Args:
        logits_tuple: Tuple of (B, K_i) logit tensors from teacher on clean unlabeled data.
        thresholds: Current per-attribute thresholds dict, mapping attr_idx -> float.
        ema_momentum: EMA momentum (α in Eq. 5 of MC-PanDA++). Higher = slower updates.
        aggregation_fn: Function to aggregate confidence scores per attribute (default: 75th percentile).
                       Called as aggregation_fn(confidences_array) -> scalar.

    Returns:
        Updated thresholds dict.
    """
    if aggregation_fn is None:
        aggregation_fn = lambda arr: torch.quantile(arr, 0.75).item()

    updated = {}
    for i, logits in enumerate(logits_tuple):
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1).values  # (B,)

        # Aggregate confidence for this attribute (e.g., 75th percentile)
        delta = aggregation_fn(max_probs)

        # EMA update: τ_a^n = α * τ_a^(n-1) + (1 - α) * δ_a^n
        old_thresh = thresholds.get(i, 0.0)
        new_thresh = ema_momentum * old_thresh + (1 - ema_momentum) * delta
        updated[i] = new_thresh

    return updated
```

**Notes:**
- `conf_thresh=0.0` (float): all predictions accepted (max prob always > 0) → no masking unless threshold is raised
- `conf_thresh={...}` (dict): per-attribute adaptive mode; start with all-zero dict and call `update_adaptive_thresholds` after each batch to handle class imbalance per attribute
- `ema_momentum=0.999` recommended (small updates, smooth convergence); `aggregation_fn` defaults to 75th percentile to be robust to outliers

---

## Step 3: Add `PseudoLabeledDataset` to `semisup.py`

Wraps an unlabeled dataset and overlays pre-generated pseudo-labels on the `target` field.

```python
from pathlib import Path
import numpy as np
from vidlu.data import Record


class PseudoLabeledDataset:
    """Wraps a dataset and replaces its `target` field with stored pseudo-labels.

    Args:
        dataset: Underlying unlabeled dataset (BihSequence or similar).
        pseudo_labels: Path to .npz file (with 'labels' array of shape (N, A),
                       int16, -1 for masked) or a (N, A) numpy array directly.

    The .npz is produced by tools/generate_pseudo_labels.py.
    """

    def __init__(self, dataset, pseudo_labels):
        self._dataset = dataset
        if isinstance(pseudo_labels, (str, Path)):
            npz = np.load(pseudo_labels, allow_pickle=False)
            arr = npz['labels']
            if len(arr) != len(dataset):
                raise ValueError(f"Pseudo-label count {len(arr)} != dataset size {len(dataset)}")
            self._labels = arr  # (N, A) int16
        else:
            self._labels = np.asarray(pseudo_labels)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        record = self._dataset[idx]
        pseudo_target = torch.from_numpy(self._labels[idx].astype(np.int64))
        return Record(record, target=pseudo_target)

    # Delegate other dataset attributes (subset, name, etc.) as needed
```

For offline training, the caller concatenates `labeled_dataset + PseudoLabeledDataset(unlabeled_dataset, path)` using vidlu's dataset concatenation (if `__add__` is supported) or a standard `ConcatDataset`.

---

## Step 4: Add `MultiAttributePseudoLabelStep` to `training.py`

Subclasses `SemisupCleanTargetConsStepBase`. The base class handles the full training loop; we override two methods only.

**Imports to add:**
```python
import copy
import os
from vidlu.training.steps import SemisupCleanTargetConsStepBase
```

**The class:**
```python
@dc.dataclass
class MultiAttributePseudoLabelStep(SemisupCleanTargetConsStepBase):
    """Pseudo-label self-training step for multi-attribute classification.

    Model output must be a tuple of (B, K_i) logit tensors; targets must be
    (B, A) integer tensors. Uses a frozen pre-trained teacher (loaded from
    checkpoint) to generate hard argmax pseudo-labels with per-(sample, attribute)
    confidence thresholding and temperature scaling.

    FixMatch-style: teacher runs on clean x_u, student is trained on jittered x_u
    (requires SemisupVAT(ColorJitterAttack) extension in the trainer config).
    """
    pre_trained_teacher: T.Optional[T.Union[str, os.PathLike, torch.nn.Module]] = None
    temperature: float = 1.0
    conf_thresh: float = 0.0
    eval_mode_teacher: bool = True  # always True for frozen teacher

    # Internal: teacher cached after first step (not a constructor param)
    _teacher_cache: T.ClassVar = None  # use instance attr set in get_student_and_teacher

    def get_student_and_teacher(self, trainer):
        model = trainer.model
        if not hasattr(self, '_teacher') or self._teacher is None:
            if isinstance(self.pre_trained_teacher, (str, os.PathLike)):
                path = self.pre_trained_teacher
                if isinstance(path, str) and path.startswith('$'):
                    path = os.environ[path[1:]]
                teacher = copy.deepcopy(model)
                params = torch.load(path, map_location='cpu')
                teacher.load_state_dict(params)
            elif self.pre_trained_teacher is None:
                teacher = model  # self-training: student is its own teacher
            else:
                teacher = self.pre_trained_teacher
            teacher.eval()
            teacher.requires_grad_(False)
            self._teacher = teacher
        # Ensure teacher is on same device as model
        model_dev = next(model.parameters()).device
        if next(self._teacher.parameters()).device != model_dev:
            self._teacher.to(model_dev)
        return model, self._teacher

    def _get_cons_loss_and_output_to_target(self, attack):
        from vidlu_irap_gaim.semisup import get_hard_pseudo_labels, update_adaptive_thresholds
        loss_cons = MultiAttributeCrossEntropyLoss(ignore_index=-1)

        temperature = self.temperature
        conf_thresh = self.conf_thresh  # can be float or dict
        adaptive_thresholds = {}  # track per-attribute adaptive thresholds if conf_thresh is dict

        def output_to_target(out_u):
            nonlocal adaptive_thresholds
            # Use adaptive thresholds if conf_thresh is a dict; otherwise use fixed value
            thresh_to_use = adaptive_thresholds if isinstance(conf_thresh, dict) else conf_thresh
            labels, _ = get_hard_pseudo_labels(out_u, temperature=temperature,
                                               conf_thresh=thresh_to_use)
            # Update adaptive thresholds for next iteration if in adaptive mode
            if isinstance(conf_thresh, dict):
                adaptive_thresholds = update_adaptive_thresholds(
                    out_u, adaptive_thresholds, ema_momentum=0.999
                )
            return labels

        return loss_cons, output_to_target
```

**Note on `_teacher` field:** Dataclasses require all fields to be declared. Use `_teacher: T.Optional[torch.nn.Module] = dc.field(default=None, repr=False, compare=False)` (not a ClassVar) to make it a proper instance field that won't cause issues with `__init__` signature.

**Adaptive threshold usage:** To enable per-attribute adaptive thresholding (recommended), pass a dict for `conf_thresh` in the trainer config, e.g.:
```python
MultiAttributePseudoLabelStep(
    pre_trained_teacher='/path/to/ckpt.pth',
    conf_thresh={i: 0.0 for i in range(41)},  # start with zeros, adapt per iteration
    temperature=1.0,
    alpha=1.0,
    amp=True,
)
```

---

## Step 5: Add trainer configs to `training.py`

**On-the-fly trainer (FixMatch-style):**
```python
irap_pseudo_label_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=True),
    train_step=MultiAttributePseudoLabelStep(
        conf_thresh=0.0,    # override per experiment
        temperature=1.0,    # override per experiment
        alpha=1.0,
        amp=True,
    ),
    loss=MultiAttributeCrossEntropyLoss(),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        partial(SemisupVAT, attack_f=partial(ColorJitterAttack, preset=JITTER_STRONG)),
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        _make_dynamic_balanced_recall_weights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)
```

`pre_trained_teacher` is set at experiment time, e.g.:
```python
TrainerConfig(irap_pseudo_label_trainer,
              train_step=MultiAttributePseudoLabelStep(pre_trained_teacher='/path/to/ckpt.pth',
                                             conf_thresh=0.8, temperature=1.2, alpha=1.0, amp=True))
```

**Offline trainer (with PseudoLabeledDataset):**
```python
irap_pseudo_label_offline_trainer = TrainerConfig(
    eval_step=SupervisedStep(eval=True, amp=True),
    train_step=SupervisedStep(amp=True),
    loss=MultiAttributeCrossEntropyLoss(ignore_index=-1),
    optimizer_f=partial(torch.optim.Adam, lr=5e-5, weight_decay=1e-3),
    epoch_count=epoch_count,
    batch_size=12,
    eval_batch_size=32,
    eval_count=epoch_count,
    jitter=make_sequence_color_jitter(),
    extension_fs=[
        partial(FreezeThenFinetune, num_frozen_epochs=2),
        _make_dynamic_balanced_recall_weights,
        VisualizationExtension,
        MultiAttributeScorePrinter,
    ],
)
```

The data passed to this trainer should be `labeled_dataset + PseudoLabeledDataset(unlabeled, path)`.

---

## Step 6: Add `vidlu_irap_gaim/tools/generate_pseudo_labels.py`

Standalone script for offline pseudo-label generation. Intended for programmatic use (model object passed in); CLI stub provided.

Key function signature:
```python
def generate_pseudo_labels(
    model: torch.nn.Module,
    dataset,
    conf_thresh: float = 0.0,
    temperature: float = 1.0,
    batch_size: int = 32,
    device: str = 'cuda',
) -> dict:
    """
    Run model on dataset, return dict with:
        'labels':       (N, A) int16 array, -1 for low-confidence
        'confidences':  (N, A) float32 array, max softmax prob per (sample, attr)
        'segment_ids':  (N,) str array (optional, if dataset provides segment_id)
    Prints coverage: fraction of (sample, attr) pairs with valid pseudo-labels.
    """
```

Saves via `np.savez_compressed(output_path, **result)`.

Model construction is done outside this function (using vidlu factories or manual instantiation) and passed in — the script does not need to know the model architecture.

---

## Implementation Order

1. `losses.py` — `ignore_index` support (no dependencies)
2. `semisup.py` — `get_hard_pseudo_labels` (no dependencies)
3. `semisup.py` — `PseudoLabeledDataset` (depends on step 2 for imports, but logic independent)
4. `training.py` — `MultiAttributePseudoLabelStep` (depends on steps 1 & 2)
5. `training.py` — trainer configs (depends on step 4)
6. `tools/generate_pseudo_labels.py` — new file (depends on step 2)

Steps 1 & 2 can be done in parallel; steps 3 & 4 can be done in parallel after step 2.

---

## Verification

1. **Unit test `get_hard_pseudo_labels` (fixed mode)**: construct a tuple of random logits, check shape `(B, A)`, verify `-1` appears when `conf_thresh` is high, check mask alignment.
2. **Unit test `get_hard_pseudo_labels` (adaptive mode)**: pass logits and a dict `conf_thresh`, verify per-attribute masking respects attribute-specific thresholds.
3. **Unit test `update_adaptive_thresholds`**: verify EMA update formula, check that thresholds converge smoothly, test with different aggregation functions (mean, median, quantile).
4. **Unit test `MultiAttributeCrossEntropyLoss(ignore_index=-1)`**: pass targets with some `-1` entries, verify loss is a valid scalar (not nan) even when a full attribute column is all `-1`.
5. **`MultiAttributePseudoLabelStep` smoke test**: instantiate with `pre_trained_teacher=None` (self-training mode), run one step with fixed `conf_thresh=0.5`, verify `loss_l` and `loss_u` are finite scalars.
6. **`MultiAttributePseudoLabelStep` adaptive test**: run one step with `conf_thresh={i: 0.0 for i in range(41)}`, verify adaptive thresholds update correctly across iterations.
7. **End-to-end**: run `irap_pseudo_label_trainer` for 2 epochs on a small dataset subset with a checkpoint from a previous run, both in fixed and adaptive modes.
8. **Offline path**: run `generate_pseudo_labels`, inspect the `.npz`, construct `PseudoLabeledDataset`, verify `target` field shape matches `(A,)` with dtype `int64`.
