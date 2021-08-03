#!/bin/bash

set -e

cd "$(pwd | grep -o '.*/scripts')"  # moves to the directory that contains the run.py script

# Table 1: Experiments on half-resolution Cityscapes with different proportions of labels ##########
start_seed=53

# DeepLab-v2-RN101

name="DL supervised"
printf "\n${name}\n"

let seed=start_seed
for divisor in 8 4 2 1; do  # How many times the labeled dataset is smaller than the whole dataset.
  # An epoch corresponds to one pass through labeled data. When not all labeled examples are used, more epochs are needed to get the same number of iterations.
  let epoch_count=divisor*100  # 800, 400, 200, 100 (always 74400 iterations)
  printf "\n${name}: 1/$divisor labeled examples, $epoch_count epochs\n\n"
  python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=4" --params "id:->backbone:deeplabv1_resnet101-coco" --no_init_eval --resume_or_start
done

name="DL MT-PhTPS"
printf "\n${name}\n"

let seed=start_seed
for divisor in 8 4 2 1; do
  let epoch_count=divisor*100  # 800, 400, 200, 100
  printf "\n${name}: 1/$divisor labeled examples, $epoch_count epochs\n\n"
  python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5,mem_efficient=True),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[4,4]" --params "id:->backbone:deeplabv1_resnet101-coco" --no_init_eval --resume_or_start
done

name="DL MT-CutMix with L2 loss, confidence thresholding"
printf "\n${name}\n"

let seed=start_seed
for divisor in 8 4 2 1; do
  let epoch_count=divisor*100  # 800, 400, 200, 100
  printf "\n${name}: 1/$divisor labeled examples, $epoch_count epochs\n\n"
  python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(loss_cons=partial(losses.conf_thresh_probs_sqr_l2_dist_ll,conf_thresh=0.97)),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[4,4]" --params "id:->backbone:deeplabv1_resnet101-coco" --no_init_eval --resume_or_start
done

# SwiftNet-RN18

name="SN supervised"
printf "\n${name}\n"

for s in {0..0}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200 (always 74400 iterations)
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=8" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done

name="SN simple-PhTPS"
printf "\n${name}\n"

for s in {0..4}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.SemisupVATTrainStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done

name="SN simple-CutMix"
printf "\n${name}\n"

for s in {0..4}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.SemisupVATTrainStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done

name="SN MT-PhTPS"
printf "\n${name}\n"

for s in {0..4}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done

name="SN MT-CutMix"
printf "\n${name}\n"

for s in {0..4}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done

name="SN MT-CutMix with L2 loss, confidence thresholding"
printf "\n${name}\n"

for s in {0..4}; do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    printf "\n${name}: seed=$seed, 1/$divisor labeled examples, $epoch_count epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(loss_cons=partial(losses.conf_thresh_probs_sqr_l2_dist_ll,conf_thresh=0.97)),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet:backbone->backbone.backbone:resnet18" --resume_or_start
  done
done
