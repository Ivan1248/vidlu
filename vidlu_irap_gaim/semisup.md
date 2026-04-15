# Semi-supervised Training Commands

All commands use the same total number of training iterations as the supervised baseline
(`make_bih_data()`, `epoch_count=10`). Since epoch length is bounded by the labeled split,
the required epoch count scales as `10 / labeled_ratio`.

| labeled_ratio | labeled_count | epoch_count |
|---|---|---|
| 1/128 | 1636 | 1280 |
| 1/64 | 3273 | 640 |
| 1/16 | 13091 | 160 |
| 1/1 | 209459 | 10 |

`eval_count=10` is set explicitly to keep evaluation frequency reasonable.
All trainers use the `_nofreeze` variant — no backbone freezing phase.

---

## Supervised baseline

```bash
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home python scripts/run.py train \
  "irap_gaim.make_bih_data()" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_local_rec_trainer_nofreeze" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()"
```

---

## Consistency (Ph20)

```bash
# 1/128 (1636 labeled)
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=1/128,use_all_as_unlabeled=True,shuffle=False)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_semisup_trainer_ph20_nofreeze,epoch_count=1280,eval_count=10" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" -e 0

# 1/64 (3273 labeled)
CUDA_VISIBLE_DEVICES=4 IRAP_HOME=~/projects/irap_home python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=1/64,use_all_as_unlabeled=True,shuffle=False)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_semisup_trainer_ph20_nofreeze,epoch_count=640,eval_count=10" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" -e 0

# 1/16 (13091 labeled)
CUDA_VISIBLE_DEVICES=4 IRAP_HOME=~/projects/irap_home python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=1/16,use_all_as_unlabeled=True,shuffle=False)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_semisup_trainer_ph20_nofreeze,epoch_count=160,eval_count=10" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" -e 0

# 1/1 (209459 labeled)
CUDA_VISIBLE_DEVICES=4 IRAP_HOME=~/projects/irap_home python scripts/run.py train \
  "irap_gaim.make_semisup_bih_data(labeled_ratio=1/1,use_all_as_unlabeled=True,shuffle=False)" \
  "standardize" \
  "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" \
  "irap_gaim.irap_semisup_trainer_ph20_nofreeze,epoch_count=10,eval_count=10" \
  --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" \
  --metrics "irap_gaim.get_irap_metrics()" -e 0
```

---

## Pseudo-label (on-the-fly)

Teacher is trained on the same labeled subset. Student uses on-the-fly pseudo-labels from the
frozen teacher checkpoint.

```bash
# 1/128 (1636 labeled)
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home \
python scripts/run_pseudolabel_pipeline.py \
  --data-supervised "irap_gaim.make_semisup_bih_data(labeled_ratio=1/128,shuffle=False)" \
  --data-semisup "irap_gaim.make_semisup_bih_data(labeled_ratio=1/128,use_all_as_unlabeled=True,shuffle=False)" \
  --trainer-supervised "irap_gaim.irap_local_rec_trainer_nofreeze,epoch_count=1280,eval_count=10" \
  --trainer-pseudolabel "irap_gaim.irap_pseudo_label_trainer_nofreeze,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='{teacher_path}',conf_thresh={conf_thresh},temperature={temperature}),epoch_count=1280,eval_count=10" \
  --metrics "irap_gaim.get_irap_metrics()" \
  --name "pseudolabel_r128" --mode onthefly

# 1/64 (3273 labeled)
CUDA_VISIBLE_DEVICES=4 IRAP_HOME=~/projects/irap_home \
python scripts/run_pseudolabel_pipeline.py \
  --data-supervised "irap_gaim.make_semisup_bih_data(labeled_ratio=1/64,shuffle=False)" \
  --data-semisup "irap_gaim.make_semisup_bih_data(labeled_ratio=1/64,use_all_as_unlabeled=True,shuffle=False)" \
  --trainer-supervised "irap_gaim.irap_local_rec_trainer_nofreeze,epoch_count=640,eval_count=10" \
  --trainer-pseudolabel "irap_gaim.irap_pseudo_label_trainer_nofreeze,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='{teacher_path}',conf_thresh={conf_thresh},temperature={temperature}),epoch_count=640,eval_count=10" \
  --metrics "irap_gaim.get_irap_metrics()" \
  --name "pseudolabel_r64" --mode onthefly

# 1/16 (13091 labeled)
CUDA_VISIBLE_DEVICES=4 IRAP_HOME=~/projects/irap_home \
python scripts/run_pseudolabel_pipeline.py \
  --data-supervised "irap_gaim.make_semisup_bih_data(labeled_ratio=1/16,shuffle=False)" \
  --data-semisup "irap_gaim.make_semisup_bih_data(labeled_ratio=1/16,use_all_as_unlabeled=True,shuffle=False)" \
  --trainer-supervised "irap_gaim.irap_local_rec_trainer_nofreeze,epoch_count=160,eval_count=10" \
  --trainer-pseudolabel "irap_gaim.irap_pseudo_label_trainer_nofreeze,train_step=irap_gaim.MultiAttributePseudoLabelStep(pre_trained_teacher='{teacher_path}',conf_thresh={conf_thresh},temperature={temperature}),epoch_count=160,eval_count=10" \
  --metrics "irap_gaim.get_irap_metrics()" \
  --name "pseudolabel_r16" --mode onthefly
```

---

## Notes

- `-r ?` can be added to any pipeline command to continue an interrupted run; `-r restart` deletes existing checkpoints and restarts from scratch.
- `--phases supervised` runs only the teacher training, useful for reusing one teacher
  across multiple student experiments.
