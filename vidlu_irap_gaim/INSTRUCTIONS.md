# IRAP GAIM ViDLU Extension - Usage Instructions

This document provides step-by-step instructions for using the `vidlu_irap_gaim` extension to reproduce the IRAP GAIM local recognition experiments.

## Prerequisites

1. **Data Setup**: Set the `IRAP_HOME` environment variable to point to a directory containing:
   - `IRAP_BIH/` - Directory with RGB images (and optionally depth maps)
   - `IRAP_BIH_METADATA/` - Directory with JSON metadata files:
     - `splits.json` - Train/val/test splits
     - `segment_id_to_data_paths_rel.json` - Mapping from segment IDs to image paths
     - `segment_id_to_road_data.json` - Road attribute data per segment
     - `attribute_metadata.json` - Attribute definitions and value mappings
     - `road_id_to_segment_id_sequence.json` - Road sequences for context windows

2. **Vistas Pretraining (Optional)**: If using Vistas pretraining, place `vistas.pt` at:
   - `$IRAP_HOME/weights/vistas.pt` (recommended), or
   - Your ViDLU pretrained directory (set via `VIDLU_PRETRAINED` or `dirs.pretrained`)

## Basic Training (Reproducing `train_local_rec_paper.sh`)

### Step 1: Prepare the Command

The basic command structure is:

```bash
python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,pretrained_backbone=True" \
  "irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" \
  -e 1
```

### Step 2: Understanding the Components

- **Data**: `irap_gaim.make_bih_data()` creates train/val/test datasets from IRAP metadata
- **Model**: `ImageSequenceClassifier` with ResNet18+SPP backbone and multi-attribute heads
- **Trainer**: `irap_local_rec_trainer` includes:
  - 2 frozen epochs + 8 finetune epochs (total 10 epochs)
  - Dynamic balanced recall loss (automatically filters to `attrs_to_include`)
  - Batch size 12 (train) / 32 (eval)
- **Metrics**: Macro F1 and accuracy over the canonical `attrs_to_include` subset (40 attributes)

### Step 3: Configure Metrics with Attribute Filtering

> [!important] **Metrics configuration**: The `MultiAttributeMacroF1` and `MultiAttributeAccuracy` metrics **require** `attrs_idx` to be set. Use the `get_irap_metrics()` helper to automatically configure metrics with `attrs_to_include` filtering.

**Recommended approach** (uses helper):

```bash
python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,pretrained_backbone=True" \
  "irap_local_rec_trainer=irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" \
  -e 1
```

**Alternative approach** (manual configuration):

```python
# In a Python script or factory expression:
from vidlu_irap_gaim import make_bih_data, get_attrs_to_include, map_attr_names_to_indices, MultiAttributeMacroF1, MultiAttributeAccuracy

datasets = make_bih_data()
train_ds = datasets['train']
attr_names = get_attrs_to_include()
attrs_idx = map_attr_names_to_indices(attr_names, train_ds.info.attribute_names)

metrics = [
    MultiAttributeMacroF1(class_counts=train_ds.info.class_counts, attrs_idx=attrs_idx),
    MultiAttributeAccuracy(attrs_idx=attrs_idx),
]
```

### Step 4: Run Training

Execute the command. The training will:

1. Load datasets and filter segments with valid context windows
2. Initialize model with ImageNet-pretrained ResNet18 backbone
3. Train for 2 epochs with only heads+SPP trainable (frozen phase)
4. Switch to finetune phase and train all parameters for 8 epochs
5. After each epoch, recompute class weights from validation recalls
6. Report metrics only over the `attrs_to_include` subset

## Using Vistas Pretraining

To use Vistas pretraining instead of ImageNet:

1. **Locate or download `vistas.pt`** and place it in your pretrained directory

2. **Use the pretraining helper**:
   ```bash
   python scripts/run.py train \
     "irap_gaim.make_bih_data()" \
     "id" \
     "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,pretrained_backbone=False" \
     "irap_gaim.irap_local_rec_trainer" \
     --params "identity:vistas.pt" \
     --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" \
     -e 1
   ```

   > [!note] **Note**: `pretrained_backbone=False` prevents ImageNet loading, and `--params` loads Vistas weights.

   Alternatively, use the `vistas_params_spec()` helper:
   ```bash
   --params "irap_gaim.vistas_params_spec()"
   ```

## Customizing the Training Schedule

To change the frozen/finetune epoch counts or learning rates:

```python
from vidlu_irap_gaim import irap_local_rec_trainer, FreezeThenFinetune, DynamicBalancedRecallWeights
from functools import partial
from vidlu.configs.training import TrainerConfig

custom_trainer = TrainerConfig(
    **irap_local_rec_trainer.normalized(),  # Copy base config
    epoch_count=15,  # Change total epochs
    extension_fs=[
        partial(FreezeThenFinetune, frozen_epochs=3, finetune_epochs=12, ...),  # Custom schedule
        DynamicBalancedRecallWeights,
    ],
)
```

## Using All Attributes (Not Just `attrs_to_include`)

By default, the trainer and metrics filter to the canonical 40-attribute subset. To use all attributes:

1. **Create metrics without filtering** (pass `attrs_idx` with all attribute indices):
   ```python
   from vidlu_irap_gaim import MultiAttributeMacroF1, MultiAttributeAccuracy, make_bih_data
   
   datasets = make_bih_data()
   train_ds = datasets['train']
   # Get all attribute indices
   all_attrs_idx = list(range(len(train_ds.info.attribute_names)))
   
   metrics = [
       MultiAttributeMacroF1(class_counts=train_ds.info.class_counts, attrs_idx=all_attrs_idx),
       MultiAttributeAccuracy(attrs_idx=all_attrs_idx),
   ]
   ```

2. **Create a custom trainer** with `DynamicBalancedRecallWeights` configured for all attributes:
   ```python
   from vidlu_irap_gaim import DynamicBalancedRecallWeights, make_bih_data
   
   datasets = make_bih_data()
   train_ds = datasets['train']
   all_attrs_idx = list(range(len(train_ds.info.attribute_names)))
   
   custom_trainer = TrainerConfig(
       **irap_local_rec_trainer.normalized(),
       extension_fs=[
           FreezeThenFinetune(...),
           partial(DynamicBalancedRecallWeights, attrs_idx=all_attrs_idx),
       ],
   )
   ```

> [!warning] **Note**: `DynamicBalancedRecallWeights` will automatically use `attrs_to_include` if `attrs_idx` is `None` and the dataset has `info.attribute_names`. To use all attributes, you must explicitly pass `attrs_idx` with all indices.

## Feature Export for Sequential Enhancement

After training, export features for sequential enhancement:

```bash
python scripts/run.py test \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False" \
  "irap_gaim.irap_local_rec_trainer" \
  -r best \
  -m "irap_gaim:export_feats,split='train',feat_dir='FEATS/train'"
```

This creates `.npy` files (one per segment) in `FEATS/train/`.

## Troubleshooting

### Error: "DynamicBalancedRecallWeights requires dataset.info.attribute_names"

**Cause**: The dataset doesn't have `info.attribute_names` set.

**Solution**: Ensure you're using `make_bih_data()` which automatically sets this, or manually set `dataset.info.attribute_names` to the ordered list of attribute names.

### Error: "MultiAttributeMacroF1.attrs_idx must be set"

**Cause**: Metrics were created without `attrs_idx`.

**Solution**: Use `get_irap_metrics(dataset)` helper, or explicitly pass `attrs_idx` when creating metrics.

### Error: "Attribute 'X' not found in dataset attributes"

**Cause**: The `attrs_to_include` list contains an attribute name that doesn't exist in the dataset's metadata.

**Solution**: Check that your `attribute_metadata.json` contains all attributes in the canonical `attrs_to_include` list. Or use a custom subset that matches your metadata.

### Error: "vistas.pt not found"

**Cause**: Vistas weights file is missing or not in expected location.

**Solution**: 
1. Download `vistas.pt` from the original IRAP GAIM repository
2. Place it at `$IRAP_HOME/weights/vistas.pt` (requires `IRAP_HOME` environment variable)
3. Or provide explicit path: `vistas_params_spec(vistas_weights_path='/path/to/vistas.pt')`

## Advanced: Custom Attribute Subsets

To use a custom subset of attributes:

```python
from vidlu_irap_gaim import get_attrs_to_include, map_attr_names_to_indices, MultiAttributeMacroF1, make_bih_data

# Get canonical subset
canonical = get_attrs_to_include()

# Create custom subset (e.g., first 20 attributes)
custom_subset = canonical[:20]

# Map to indices
datasets = make_bih_data()
attrs_idx = map_attr_names_to_indices(custom_subset, datasets['train'].info.attribute_names)

# Create metrics with custom subset
metrics = [
    MultiAttributeMacroF1(class_counts=datasets['train'].info.class_counts, attrs_idx=attrs_idx),
    MultiAttributeAccuracy(attrs_idx=attrs_idx),
]
```

## Reproducing Exact Paper Results

To match `train_local_rec_paper.sh` exactly:

1. **Use Vistas pretraining** (see [[#Using Vistas Pretraining]] above)
2. **Use default trainer config** (`irap_local_rec_trainer`) - it already matches the paper hyperparameters
3. **Use `attrs_to_include` filtering** - this is automatic with the default setup
4. **Set deterministic mode** for reproducibility:
   ```bash
   python scripts/run.py train ... --deterministic --seed 123
   ```

## File Structure Reference

- `attrs.py` - Canonical `attrs_to_include` list and mapping helpers
- `datasets.py` - `BihSequence` dataset and `make_bih_data()` factory
- `models.py` - `ImageSequenceClassifier` model
- `losses.py` - `MultiAttributeCrossEntropyLoss` (stateful loss for dynamic weights) and `multi_attribute_cross_entropy`
- `metrics.py` - `MultiAttributeMacroF1`, `MultiAttributeAccuracy`, `get_irap_metrics()`
- `training.py` - `FreezeThenFinetune` extension and `irap_local_rec_trainer` config
- `dynamic.py` - `DynamicBalancedRecallWeights` extension
- `pretraining.py` - `vistas_params_spec()` helper
- `resnet_backbone.py` - Exact port of IRAP ResNet18+SPP architecture

## Additional Resources

- See [[.devdocs/irap_gaim_discrepancies.md]] for detailed comparison with original implementation
- See [[README.md]] for API reference
- See `libs/irap_gaim-main/train_local_rec.py` for original training script reference
