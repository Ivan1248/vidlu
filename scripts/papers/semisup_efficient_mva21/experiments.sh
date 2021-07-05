exit

# Table 1: Experiments on half-resolution Cityscapes ###############################################

start_seed=53

# DeepLab-v2-RN101

# DL supervised

seed=start_seed
for divisor in 8 4 2 1  # How many times the labeled dataset is smaller than the whole dataset.
do
  # An epoch corresponds to one pass through labeled data. When not all labeled examples are used, more epochs are needed to get the same number of iterations.
  let epoch_count=divisor*100  # 800, 400, 200, 100
  echo 1/$divisor labeled examples, ${epoch_count} epochs

  CUDA_VISIBLE_DEVICES=0 python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=4" --params "id->backbone:deeplabv1_resnet101-coco.pth" --no_init_eval
done

# DL MT-PhTPS

seed=start_seed
for divisor in 8 4 2 1
do
  let epoch_count=divisor*100  # 800, 400, 200, 100
  echo 1/$divisor labeled examples, ${epoch_count} epochs

  CUDA_VISIBLE_DEVICES=0 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5,mem_efficient=True),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[4,4]" --params "id->backbone:deeplabv1_resnet101-coco.pth" --no_init_eval
done

# DL MT-CutMix with L2 loss, confidence thresholding

seed=start_seed
for divisor in 8 4 2 1
do
  let epoch_count=divisor*100  # 800, 400, 200, 100
  echo 1/$divisor labeled examples, ${epoch_count} epochs

  CUDA_VISIBLE_DEVICES=2 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),${divisor})[0],d[0],d[1])" "standardize(cityscapes_mo)" "SegmentationModel,init=None,backbone_f=partial(vmo.deeplab.DeepLabV2_ResNet101,n_classes=19),head_f=vmc.ResizingHead" "tc.deeplabv2_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(loss_cons=partial(losses.conf_thresh_probs_sqr_l2_dist_ll,conf_thresh=0.97)),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[4,4]" --params "id->backbone:deeplabv1_resnet101-coco.pth" --no_init_eval
done

# SwiftNet-RN18

# SN supervised

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=1 python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done

# SN simple-PhTPS

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=0 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.SemisupVATTrainStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done

# SN simple-CutMix

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=0 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.SemisupVATTrainStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done

# SN MT-PhTPS

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=0 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done

# SN MT-CutMix

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=0 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done

# SN MT-CutMix with L2 loss, confidence thresholding

for s in {0..4}
do
  let seed=start_seed+s
  for divisor in 8 4 2 1
  do
    let epoch_count=divisor*200  # 1600, 800, 400, 200
    echo seed=$seed, 1/$divisor labeled examples, $epoch_count epochs
    CUDA_VISIBLE_DEVICES=1 python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),${divisor})[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_cutmix,train_step=ts.MeanTeacherStep(loss_cons=partial(losses.conf_thresh_probs_sqr_l2_dist_ll,conf_thresh=0.97)),lr_scheduler_f=lr.QuarterCosLR,epoch_count=${epoch_count},batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done


# Table 2: Consistency variant comparison ##########################################################

# supervised, simple-PhTPS and MT-PhTPS Cityscapes

1w_ps="train_step=ts.SemisupVATTrainStep(alpha=0.5)"
1w_pt="train_step=ts.SemisupVATTrainStep(alpha=0.5,rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
2w_ps="train_step=ts.SemisupVATTrainStep(alpha=0.5,rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"
2w_p2="train_step=ts.SemisupVATTrainStep(alpha=0.5,rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"

1w_ps_mt="train_step=ts.MeanTeacherStep(alpha=0.5)"
1w_pt_mt="train_step=ts.MeanTeacherStep(alpha=0.5,rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
2w_ps_mt="train_step=ts.MeanTeacherStep(alpha=0.5,rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"

for s in {0..4}
do
  let seed=start_seed+s

  echo simple-CS-1/4 supervised: seed=$seed, 800 epochs
  python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),4)[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

  for args in 1w_ps 1w_pt 2w_ps 2w_p2 1w_ps_mt 1w_pt_mt 2w_ps_mt
  do
    echo simple-CS-1/4 $args: seed=$seed, 800 epochs
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(${seed}}),4)[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,${!args},lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
  done
done