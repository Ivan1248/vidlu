#!/bin/bash

set -e

# Table 2: Consistency variant comparison ##########################################################

start_seed=53
run_count=5

name="Half-resolution Cityscapes (simple-PhTPS, MT-PhTPS)"
printf "\n${name}\n"

simp_1w_ps="ts.SemisupVATTrainStep(alpha=0.5)"
simp_1w_pt="ts.SemisupVATTrainStep(alpha=0.5,rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
simp_2w_ps="ts.SemisupVATTrainStep(alpha=0.5,rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"
simp_2w_p2="ts.SemisupVATTrainStep(alpha=0.5,block_grad_on_pert=False,block_grad_on_clean=False)"

mt_1w_ps="ts.MeanTeacherStep(alpha=0.5)"
mt_1w_pt="ts.MeanTeacherStep(alpha=0.5,rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
mt_2w_ps="ts.MeanTeacherStep(alpha=0.5,rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"

for s in {0..4}; do
  let seed=start_seed+s

  printf "\n${name}: simple-CS-1/4 supervised: seed=$seed, 800 epochs\n\n"
  python run.py train "train,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),4)[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" --resume_or_start

  for args in simp_1w_ps simp_1w_pt simp_2w_ps simp_2w_p2 mt_1w_ps mt_1w_pt mt_2w_ps
  do
    printf "\n${name}: simple-CS-1/4 $args: seed=$seed, 800 epochs\n\n"
    python run.py train "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(${seed}),4)[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=${!args},lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" --resume_or_start
  done
done


name="CIFAR-10 (simple-PhTPS, MT-PhTPS)"
printf "\n${name}\n"

simp_1w_ps="ts.SemisupVATTrainStep()"
simp_1w_pt="ts.SemisupVATTrainStep(rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
simp_2w_ps="ts.SemisupVATTrainStep(rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"
simp_2w_p2="ts.SemisupVATTrainStep(block_grad_on_pert=False,block_grad_on_clean=False)"

mt_1w_ps="ts.MeanTeacherStep()"
mt_1w_pt="ts.MeanTeacherStep(rev_cons=True,block_grad_on_pert=True,block_grad_on_clean=False)"
mt_2w_ps="ts.MeanTeacherStep(rev_cons=True,block_grad_on_pert=False,block_grad_on_clean=False)"

for f in {0..4}; do
  let fold_start=f*4000
  let fold_end=fold_start+4000

  printf "\n${name}: CIFAR-10-4000 supervised: fold $f, 1000 epochs\n\n"
  python run.py train "train,test:Cifar10{trainval,test}:(rotating_labels(d[0])[${fold_start}:${fold_end}],d[1])" id "WRN,backbone_f=t(depth=28,width_factor=2,small_input=True)" "tc.wrn_cifar,epoch_count=1000,batch_size=128,eval_batch_size=640" --resume_or_start

  for args in simp_1w_ps simp_1w_pt simp_2w_ps simp_2w_p2 mt_1w_ps mt_1w_pt mt_2w_ps
  do
    printf "\n${name}: CIFAR-10-4000 $args: seed=$seed, 1000 epochs\n\n"
    python run.py train "train,train_u,test:Cifar10{trainval,test}:(rotating_labels(d[0])[${fold_start}:${fold_end}],d[0],d[1])" id "WRN,backbone_f=t(depth=28,width_factor=2,small_input=True)" "tc.wrn_cifar,tc.semisup_cons_phtps20,train_step=${!args},epoch_count=1000,batch_size=[128,512],eval_batch_size=640" --resume_or_start
  done
done
