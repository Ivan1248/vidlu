# vidlu_irap_gaim

ViDLU extension re-implementing the IRAP GAIM local attribute recognition workflow on road segments.

This extension provides:

- Dataset construction from IRAP/BiH metadata with context windows
- A sequence model (ResNet18 backbone + SPP + multi-attribute heads, optional attention). SPP matches the original per-frame feature size (2100 with grids [6,3,2,1]). Heads are sized to the sequence length times the SPP output, as in the original implementation.
- A two-phase training schedule (frozen then finetune)
- Multi-attribute losses and metrics (macro-F1, accuracy)
- Attribute subset filtering (`attrs_to_include`) matching the paper experiments

It is designed to reproduce the original pipeline modulo randomness, using the same data selection and similar preprocessing.

> [!note] **📖 For detailed usage instructions, see [[INSTRUCTIONS.md]]**

## Install/use

No separate install required if the repository root is on `PYTHONPATH`. The extension is auto-discovered as `irap_gaim` by ViDLU.

## Data layout and configuration

This extension no longer needs a `config.json`. It uses a ViDLU-style configuration:

- Set `IRAP_HOME` to the IRAP root directory containing `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (JSONs), or pass explicit directories via factory args.
- All other options are provided as function arguments in expressions.

## Examples (training/eval)

### Basic training (matches `train_local_rec_paper.sh` with ImageNet pretraining)

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,pretrained_backbone=True" \
  "irap_gaim.irap_local_rec_trainer" \
  --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" \
  -e 1
```

> [!important] **Metrics configuration**: The `MultiAttributeMacroF1` and `MultiAttributeAccuracy` metrics **require** `attrs_idx` to be set. Use the `get_irap_metrics()` helper to automatically configure metrics with `attrs_to_include` filtering.

### With Vistas pretraining (requires vistas.pt in pretrained directory)

```bash
IRAP_HOME=/path/to/IRAP_HOME python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,pretrained_backbone=False" \
  "irap_gaim.irap_local_rec_trainer" \
  --params "identity:vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" \
  -e vistas1
```

> [!note] **Attribute filtering**: The default trainer config (`irap_local_rec_trainer`) uses `attrs_to_include` filtering (canonical paper subset of 40 attributes). `DynamicBalancedRecallWeights` automatically filters to these attributes. Metrics must be configured with `attrs_idx` (via `get_irap_metrics()` helper). To use all attributes, create a custom trainer config without attribute filtering.

### Export features for sequential enhancement (optional)

```bash
IRAP_HOME=/path/to/IRAP_HOME PRECOMPUTED_FEATURES_DIR=/path/to/precomputed_features \
python scripts/run.py test \
  "irap_gaim.make_bih_data()" \
  "id" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False" \
  "irap_gaim.irap_local_rec_trainer" \
  -m "irap_gaim:export_feats,split='val',feat_dir='$PRECOMPUTED_FEATURES_DIR/val'"
```
The `PRECOMPUTED_FEATURES_DIR` env var matches the `precomputed_features_dir` key in `libs/irap_gaim-main/config.json`; both can be overridden as needed.

## Sequential enhancement (optional)

After exporting features, train an LSTM enhancement model per attribute using the feature sequences (no config.json needed):

```bash
python scripts/run.py train \
  "irap_gaim.make_seq_enh_data(feat_dir='$PRECOMPUTED_FEATURES_DIR/train',attribute=0)" \
  "id" \
  "irap_gaim.GeneralLSTMModel,input_dim=<FEATURE_DIM>,n_classes=<N_CLASSES>,hidden_dim=64,n_layers=2" \
  "ct.classification" \
  --metrics "A"
```

**Notes**:
- Replace `<FEATURE_DIM>` with the flattened per-sample feature dim (inspect any `.npy` from FEATS), and `<N_CLASSES>` with the class count for the chosen attribute.
- Repeat across attributes as needed.
- `make_seq_enh_data` resolves metadata via `IRAP_HOME/IRAP_BIH_METADATA` or an explicit `metadata_dir=`.

## Data configuration

- `make_bih_data` will use `IRAP_HOME/IRAP_BIH` and `IRAP_HOME/IRAP_BIH_METADATA` if `dataset_dir`/`metadata_dir` are not provided.
- To be explicit, pass both:
  - Data: `"irap_gaim.make_bih_data(dataset_dir='/path/to/IRAP_BIH',metadata_dir='/path/to/IRAP_BIH_METADATA')"`
  - This ensures no reliance on environment variables.

**Notes**:
- `irap_gaim.irap_local_rec_trainer` reproduces a two-stage schedule: freeze then finetune.
- The default transforms mimic color jitter on train, center crop to `input_dim`, then normalization.
- Metrics compute macro F1 and attribute-wise accuracy over `attrs_to_include` subset.
- Dynamic class weighting strictly requires a validation split (prefix `val`) and dataset `info.class_counts` and `info.attribute_names`.

## Components

### Data factories

- `irap_gaim.make_bih_data(dataset_dir=None, metadata_dir=None, context_sequence=(0,-1,-4), data_types=("rgb",), mean=..., std=..., input_dim_rgb=(384,288,3), attribute_value_mapping_path=None, transforms=None, label_map=None, ncontext_segment_id_subset=None)`
  - Builds `{train,val,test}` datasets using `IRAP_HOME` for paths if not provided.
  - Supports `ncontext_segment_id_subset` parameter for filtering segments globally across all splits.

- `irap_gaim.get_class_counts(metadata_dir=None, attribute_metadata_path=None, attribute_value_mapping_path=None)`
  - Computes number of classes per attribute (after optional value mapping).

- `irap_gaim.BihSequence(dataset_dir, metadata_dir, ...)`
  - Reads metadata, filters to segments with full context, outputs sequences and targets.

### Models

- `irap_gaim.ImageSequenceClassifier(class_counts, attention=False, pretrained_backbone=True)`
  - ResNet18 backbone (exact port from original), SPP, optional per-attribute attention; one linear head per attribute.
  - Implements `get_trainable_parameters()` method for frozen-phase training.

### Training

- `irap_gaim.irap_local_rec_trainer`
  - `TrainerConfig` using supervised step, `MultiAttributeCrossEntropyLoss`, `FreezeThenFinetune` (2 frozen + 8 finetune epochs), and `DynamicBalancedRecallWeights`.
  - Defaults match `train_local_rec_paper.sh`.
  - `DynamicBalancedRecallWeights` automatically uses `attrs_to_include` filtering when `attrs_idx` is `None`.

### Attribute filtering helpers

- `irap_gaim.get_attrs_to_include()` → Returns canonical paper subset tuple (40 attributes)
- `irap_gaim.map_attr_names_to_indices(attr_names, dataset_attribute_names)` → Maps attribute names to indices
- `irap_gaim.ATTRS_TO_INCLUDE` → Canonical tuple constant

### Losses and metrics

- `MultiAttributeCrossEntropyLoss()` → Stateful callable that supports attribute subset filtering plus dynamic per-class weights (required for `DynamicBalancedRecallWeights`)
- `multi_attribute_cross_entropy(outputs, targets, attrs_idx=None)` → Stateless helper for manual usage without dynamic weighting
- `MultiAttributeMacroF1(class_counts, attrs_idx)` → **Requires** `attrs_idx` (raises `RuntimeError` if `None`)
- `MultiAttributeAccuracy(attrs_idx)` → **Requires** `attrs_idx` (raises `RuntimeError` if `None`)
- `get_irap_metrics(dataset, class_counts=None, attrs_to_include=None)` → Helper that automatically configures metrics with `attrs_to_include` filtering (defaults to canonical paper subset)

### Feature export and pretraining

- `export_feats(exp, split, feat_dir)` → Exports per-segment feature `.npy` files (as a `--module` callable)
- `vistas_params_spec(vistas_weights_path=None)` → Returns path to vistas.pt for `--params`

### Sequential enhancement

- Data: `make_seq_enh_data(feat_dir, attribute, context_sequence, metadata_dir=None, irap_home=None, attribute_value_mapping_path=None)`
- Model: `GeneralLSTMModel(input_dim, n_classes, hidden_dim, n_layers, bidirectional)`

## Dynamic class weighting and attribute subset

The default `irap_local_rec_trainer` enables `DynamicBalancedRecallWeights`, which:

- Periodically recomputes per-attribute class weights from validation recalls
- Reweights cross-entropy loss per attribute
- Reproduces "dynamic balanced recall" from the original paper
- Automatically uses `attrs_to_include` filtering when `attrs_idx` is `None` (requires dataset `info.attribute_names`)

By default, training and metrics are restricted to `attrs_to_include` (canonical paper subset of 40 attributes). This matches the original `train_local_rec_paper.sh` behavior. The extension automatically:

- Filters loss computation to selected attributes (via `DynamicBalancedRecallWeights`)
- Filters dynamic weight updates to selected attributes
- Filters metric accumulation to selected attributes (via `get_irap_metrics()` helper)

To use all attributes or a custom subset, create a custom trainer config and pass `attrs_idx` to metrics/loss explicitly.

## Reproducibility

- Set `--deterministic` and seed in `scripts/run.py` for full determinism where possible.
- Data selection: we replicate filtering to segments that have a valid context window across the split, using `road_id_to_segment_id_sequence_path` and `context_sequence`.
- Preprocessing: jitter + crop + normalize to match provided `normalization_statistics`.
- Two-phase schedule mirrors original hyperparameters (can be overridden by editing `FreezeThenFinetune` init or adding a wrapper TrainerConfig).

## Known differences vs. original

See [[.devdocs/irap_gaim_discrepancies.md]] for detailed comparison.

Key differences:
- Depth/raster inputs not yet supported (RGB only)
- Sequential enhancement pipeline simplified (LSTM only, no transformer)
- Attribute value mapping segment filtering differs slightly (`_remove_filtered_out_segments` logic not fully replicated)
- Transforms always apply center crop (even when image size matches `input_dim_rgb`)

## Configuration summary

- Directories: set via args or `IRAP_HOME`.
- Pixel stats: defaults to BiH mean/std (hard-coded), can be overridden via args.
- Context: `context_sequence` argument (default `[0,-1,-4]`).

## TODOs

- Add additional models (LSTM/Transformer) for sequential smoothing (`seq_enh_model.py`).
- Add attention mode parity tests, more unit tests, and a preset launcher that encapsulates data+model+trainer+metrics.
- Add a transformer-based sequential enhancement model preset and exact parity with original feature shapes.
- Implement depth/raster input support for multi-stream models.

## Troubleshooting

- Ensure `VIDLU_DATA` and other dirs are set (see `scripts/dirs.py`) and the config paths are correct.
- If `torchvision` pretrained weights download fails, set `pretrained_backbone=False` in the model config.
- Use `--debug` to enable anomaly detection.

### Common errors

**"MultiAttributeMacroF1.attrs_idx must be set"**
- **Solution**: Use `get_irap_metrics(dataset)` helper instead of creating metrics directly.

**"DynamicBalancedRecallWeights requires dataset.info.attribute_names"**
- **Solution**: Ensure you're using `make_bih_data()` which automatically sets `info.attribute_names`.

**"vistas.pt not found"**
- **Solution**: Place `vistas.pt` at `$IRAP_HOME/weights/vistas.pt` (requires `IRAP_HOME` environment variable), or use `vistas_params_spec(vistas_weights_path='/path/to/vistas.pt')`.

## Strict requirements

- Validation data with prefix `val` must be present; otherwise training will error where dynamic weights are computed.
- Datasets must populate `info.class_counts` with per-attribute class counts.
- Datasets must populate `info.attribute_names` for automatic `attrs_to_include` filtering.
- Models used for feature export must implement `forward(..., return_features=True)`; otherwise export will error.
