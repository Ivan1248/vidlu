# IRAP GAIM Training Setup & Hyperparameters

This document details the exact hyperparameters and definitions used in the IRAP GAIM training pipeline, comparing the original implementation (`train_local_rec.py`) with the Vidlu adaptations (`hybrid_training_v3.py` and `scripts/run.py`).

## 1. Optimizer Configuration (Adam)

The training uses the Adam optimizer with standard betas `(0.9, 0.999)`.

| Phase | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| **Frozen** | Learning Rate | `5e-5` | Applied to trainable heads & SPP only |
| | Weight Decay | `1e-3` | |
| | Scheduler Gamma | `0.8` | Multiplicative decay per epoch |
| **Finetune** | Learning Rate | `1e-5` | Applied to all parameters |
| | Weight Decay | `1e-3` | |
| | Scheduler Gamma | `0.88` | Multiplicative decay per epoch |

## 2. Training Loop Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | `12` | Training batch size |
| **Eval Batch Size** | `32` | Validation/Test batch size (Original used 12) |
| **Frozen Epochs** | `2` | Number of epochs with frozen backbone |
| **Finetune Epochs** | `8` | Number of epochs with full fine-tuning |
| **Total Epochs** | `10` | Total training duration |
| **AMP** | `True` | Automatic Mixed Precision enabled |
| **Drop Last** | `True` | Drop last incomplete batch during training |

## 3. Loss Function

**Type:** `MultiAttributeCrossEntropyLoss` (Vidlu) / `F.cross_entropy` loop (Original)

The loss is calculated as the mean of cross-entropy losses across all included attributes.

**Formula:**
```python
total_loss = mean(cross_entropy(logits[attr], target[attr], weight=class_weights[attr]) for attr in attributes)
```

**Reduction:**
1.  **Per-attribute:** `mean` over the batch dimension.
2.  **Global:** `mean` over the attribute dimension.

## 4. Class Weight Calculation

Class weights are dynamic and updated after every epoch based on validation recall.

**Initial Weights (Epoch 0):**
Based on class frequency in training set, assuming a random classifier recall (`1/n_classes`).

**Updated Weights (Epoch > 0):**
Based on class frequency and actual validation recall.

**Formula:**
```python
weights = inv_freq * (1 - recall) + sqrt(inv_freq) * recall
```
Where:
*   `inv_freq = total_occurrences / class_occurrences`
*   `recall` = Per-class recall (or `1/n_classes` initially)

## 5. Data Augmentation

Applied via `ColorJitter` in the dataset transformation pipeline.

**Parameters:**
*   **Brightness:** `0.6`
*   **Contrast:** `0.3`
*   **Saturation:** `0.2`
*   **Hue:** `0.02`

**Implementation:**
*   Original: `build_rgb_transform()`
*   Vidlu: `make_bih_data(jitter=(0.6, 0.3, 0.2, 0.02))`

## 6. Model Architecture & Initialization

**Backbone:** ResNet (loaded from `vistas.pt`)
**Initialization:**
*   Backbone weights loaded from `vistas.pt`.
*   Key mapping: `frame_encoder.resnet.{k[9:]}` maps to `backbone.{k}` from checkpoint.
*   Strict loading: `False` (allows missing keys like heads).

**Freezing Logic:**
*   **Frozen Phase:** Only parameters returned by `get_trainable_parameters()` (Heads + SPP) are trainable. Backbone is frozen.
*   **Finetune Phase:** All parameters are trainable.

## 7. Scheduler Timing

*   **Type:** `MultiplicativeLR`
*   **Step Frequency:** Once per epoch.
*   **Timing:** Called after the training loop and before/after validation (consistent across implementations).

## 8. Loss Reporting

All implementations report loss averaged **over the reporting interval** (~218 batches), not cumulatively from the epoch start.

| Implementation | Metric Reset | Effect |
|----------------|-------------|--------|
| **run.py** (`ProgressMonitor`) | `reset=True` | Interval average |
| **hybrid_training_v3** | `reset=True` | Interval average |
| **Original** | Modified to `reset` | Interval average (Modified) |

> **Note:** The original `train_local_rec.py` used cumulative averaging, but has been modified to use interval-based reporting to match the Vidlu implementations.

