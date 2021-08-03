#!/bin/bash
# This runs 3 training configurations for 1 epoch for testing whether everything is set up.

set -e

cd "$(pwd | grep -o '.*/scripts')"  # moves to the directory that contains the run.py script

epoch_count=1  # set to 0 for evaluation only

# SwiftNet-RN18, half-resolution Cityscapes
printf "\n\nSwiftNet-RN18, half-resolution Cityscapes\n"
python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(53),4)[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=8" --params "resnet:backbone->backbone.backbone:resnet18" --no_train_eval

# DeepLab-v2-RN101, half-resolution Cityscapes
printf "\n\nDeepLab-v2-RN101, half-resolution Cityscapes\n"
python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(53),4)[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=4" --params "id:->backbone:deeplabv1_resnet101-coco" --no_init_eval --no_train_eval

# WRN-28-2, CIFAR-10
printf "\n\nWRN-28-2, CIFAR-10\n"
python run.py train "train,test:Cifar10{trainval,test}:(rotating_labels(d[0])[:4000],d[1])" id "WRN,backbone_f=t(depth=28,width_factor=2,small_input=True)" "tc.wrn_cifar,epoch_count=${epoch_count},batch_size=128,eval_batch_size=640" --no_train_eval