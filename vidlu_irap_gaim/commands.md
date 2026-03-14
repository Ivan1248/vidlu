# IRAP GAIM Training Commands

Data loading is deterministic (no photometric jitter). Normalization is handled by the `standardize` input adapter.
Photometric jittering (ColorJitter) is configured in the TrainerConfig (`irap_gaim.irap_local_rec_trainer`).

## ResNet encoder with pretrained Vistas weights

```bash
IRAP_HOME=~/projects/IRAP_HOME python scripts/run.py train "irap_gaim.make_bih_data()" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics()"
```

Notes:
- `--params "...:irap_gaim/vistas.pt"` resolves `vistas.pt` via your ViDLU pretrained directory (`dirs.pretrained` / `VIDLU_PRETRAINED`). Place the file at `<VIDLU_PRETRAINED>/irap_gaim/vistas.pt`.
- The dataset emits RGB in \([0,1]\), so `standardize` is recommended when using pretrained encoders.

## Semi-supervised (same dataset unlabeled) - ResNet encoder with pretrained Vistas weights

```bash
IRAP_HOME=~/projects/IRAP_HOME python scripts/run.py train "irap_gaim.make_semisup_bih_data(use_all_as_unlabeled=True)" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_semisup_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics()"
```

## DINOv2 ViT encoder

```bash
IRAP_HOME=~/projects/IRAP_HOME python scripts/run.py train "irap_gaim.make_bih_data()" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.dinov2_vit_encoder,variant='dinov2_vitb14',params_dir=dirs.pretrained)" "irap_gaim.irap_local_rec_trainer" --metrics "irap_gaim.get_irap_metrics()"
```

## Dataset viewer (Streamlit)

```bash
IRAP_HOME=~/projects/IRAP_HOME streamlit run vidlu_irap_gaim/tools/dataset_viewer.py
```

## Inference visualization (write PNGs per segment)

### Local model (Vidlu checkpoint)

Pass either:
- `--checkpoint_dir` pointing to a *numbered* Vidlu checkpoint directory (contains `model_state.pth`), or
- `--model_state_path` pointing directly to a state dict file.

```bash
IRAP_HOME=~/projects/IRAP_HOME python vidlu_irap_gaim/tools/inference_visualization.py \
  --mode local \
  --split val \
  --context_sequence "0,-1,-4" \
  --input_adapter standardize \
  --checkpoint_dir "/path/to/VIDLU_EXPERIMENTS/states/<experiment_name>/<checkpoint_id>" \
  --output_dir visualization_output \
  --limit 50 --verbose
```

### Sequential enhancement (legacy checkpoints)

Requires:
- `--seq_config_path` (legacy `config.json` used to define per-attribute LSTM hyperparams)
- `--seq_models_root` (directory holding per-attribute subdirs with `best_model_MF1.pt`)
- `--feat_dir` (directory with exported `SEGMENT_ID.npy` feature vectors)

```bash
IRAP_HOME=~/projects/IRAP_HOME python vidlu_irap_gaim/tools/inference_visualization.py \
  --mode sequential_legacy \
  --split val \
  --context_sequence "-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10" \
  --seq_config_path "/path/to/config.json" \
  --seq_models_root "/path/to/global_smoothing_saved_models/<run_dir>" \
  --feat_dir "/path/to/precomputed_features/val" \
  --output_dir visualization_output_seq \
  --limit 50 --verbose
```