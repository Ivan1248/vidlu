python run.py train "MNIST{train,val}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar"
python run.py train "whitenoise{train,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True),head_f=partial(vidlu.nn.components.classification_head,10)" "tc.resnet_cifar" "accuracy"
python run.py train "Cifar10{train,val}" id "SmallImageClassifier" "SmallImageClassifierTrainer,data_loader_f=t(num_workers=4)" "" -v 2
python run.py train "Cifar10{train,val}" id "DiscriminativeModel,backbone_f=partial(c.PreactBlock,kernel_sizes=[3,3,3],base_width=64,width_factors=[1,1,1]),init=partial(init.kaiming_resnet)" ResNetCifarTrainer ""
python run.py train "Cifar10{train,val}" standardize "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar"
python run.py train "Cifar10{train,val}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar"
python run.py train "Cifar10{train,val}" id "DenseNet,backbone_f=t(depth=121,small_input=True)" "tc.resnet_cifar"

# CIFAR
python run.py train "Cifar10{trainval,test}" id "ResNetV1,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar"
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar"
python run.py train "Cifar10{trainval,test}" id "WideResNet,backbone_f=t(depth=28,width_factor=10,small_input=True)" "tc.resnet_cifar"

python run.py train "Cifar10{trainval,test}" id "DenseNet,backbone_f=t(depth=121,small_input=True)" "tc.densenet_cifar"
## i-RevNet
python run.py train "Cifar10{trainval,test}" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(4,)*4,block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "tc.irevnet_cifar"
python run.py train "Cifar10{trainval,test}" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(2,6,6,2),block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "tc.irevnet_cifar"
python run.py train "Cifar10{trainval,test}" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(2,6,6,2),block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "tc.irevnet_hybrid_cifar"
python run.py train "train,test:Cifar10{trainval,test}:(d[0][:100],d[1][:100])" id "ResNetV2,backbone_f=t(depth=10,small_input=True)" "tc.resnet_cifar"



# Multistep step
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)"  "tc.resnet_cifar,train_step=tc.SupervisedTrainMultiStep(8),epoch_count=200/8"

# semantic segmentation

## DenseNet (pretrained on ImageNet)
python run.py train "Cityscapes(downsampling=2){train,val}" id "DenseNet,backbone_f=t(depth=121,small_input=False)" "tc.ladder_densenet,optimizer_maker=tc.FineTuningOptimizerMaker({'backbone':1/5})" --params "densenet(backbone),backbone:densenet121-a639ec97.pth"


## ResNet
python run.py train "camvid(downsampling=2){train,val}" id "ResNetV1,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes,jitter=jitter.SegRandomCropHFlip((360,480)),optimizer_maker=None" -r

## Swiftnet
python run.py train "Cityscapes{train,val}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes"
python run.py train "Cityscapes(downsampling=2){train,val}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes"
python run.py train "camvid{trainval,test}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_camvid"
# network deconvolution
python run.py train "Cityscapes{train,val}" id "SwiftNet,backbone_f=t(depth=18,small_input=False,block_f=t(norm_f=None,conv_f=partial(vm.DeconvConv,bias=True,padding='half')))" "tc.swiftnet_cityscapes"

## SwiftNet pretrained on Cityscapes
python run.py train "Cityscapes{train,val}" "standardize(mean=[.485,.456,.406],std=[.229,.224,.225])" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes" --params swiftnet:swiftnet_ss_cs_best.pt
python run.py train "Cityscapes{train,val}" "standardize(mean=[73.15/255,82.9/255,72.3/255],std=[47.67/255,48.49/255,47.73/255])" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes" --params swiftnet:rn18_single_scale/model.pt  # 75.39
## SwiftNet pretrained on ImageNet
python run.py train "CamVid{trainval,test}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_camvid" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
python run.py train "Cityscapes{train,val}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
python run.py train "Cityscapes{train,val}" "standardize(mean=[73.15/255,82.9/255,72.3/255],std=[47.67/255,48.49/255,47.73/255])" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"  # 75.39

## SegResNetV1 pretrained
python run.py train "Cityscapes{train,val}" standardize "SegResNetV1,backbone_f=t(depth=50,small_input=False)" "tc.ladder_densenet"

## SegResNetV1 pretrained
python run.py train "CamVid{trainval,test}" standardize "SegResNetV1,backbone_f=t(depth=50,small_input=False)" "tc.swiftnet_camvid,epoch_count=400,optimizer_maker=tc.FineTuningOptimizerMaker({'backbone': 0})" --params "resnet[backbone]->backbone:resnet50-19c8e357.pth"
python run.py train "CamVid{trainval,test}" standardize "SegResNetV1,backbone_f=t(depth=50,small_input=False)" "tc.swiftnet_camvid,epoch_count=400,optimizer_maker=tc.FineTuningOptimizerMaker({'backbone': 0})" --params "madrylab_resnet[backbone]->backbone:madrylab/cvrobust/resnet-50-imagenet.pt"


# Adversarial training
## ResNet
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar,tc.adversarial,attack_f=partial(tc.madry_Cifar10_attack,step_count=7),eval_attack_f=t(step_count=10,stop_on_success=True)"
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar,tc.adversarial,train_step=tc.AdversarialTrainStep(True),attack_f=partial(tc.madry_Cifar10_attack,step_count=7,stop_on_success=True),eval_attack_f=t(step_count=20)"
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "tc.resnet_cifar,tc.adversarial,attack_f=partial(tc.madry_Cifar10_attack,step_count=7),eval_attack_f=partial(tc.madry_Cifar10_attack,step_count=20),train_step=tc.AdversarialTrainMultiStep(),epoch_count=25"

## Swiftnet
python run.py train "Cityscapes{train,val}" standardize "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes,epoch_count=40,batch_size=1,tc.adversarial,train_step=tc.AdversarialTrainStep(False),attack_f=partial(tc.madry_Cifar10_attack, step_count=40, eps=48/255, clip_bounds=None)" --params swiftnet:swiftnet_ss_cs.pt

# Tent activations
python run.py train "Cifar10{trainval,test}" id "ResNetV2,backbone_f=t(depth=18,small_input=True,block_f=t(act_f=C.Tent))" "tc.wrn_cifar_tent,tc.adversarial},attack_f=attacks.DummyAttack,eval_attack_f=partial(tc.madry_Cifar10_attack,step_count=7,stop_on_success=True)"
python run.py train "mnist{trainval,test}" id "MNISTNet,backbone_f=t(act_f=C.Tent,use_bn=False)" "tc.mnistnet_tent,tc.adversarial,attack_f=attacks.DummyAttack,eval_attack_f=tc.mnistnet_tent_eval_attack"

# show_summary.py
python show_summary.py /home/igrubisic/data/states/Cifar10\{trainval\,test\}/ResNetV2\,backbone_f\=t\(depth\=18\,small_input\=True\)/Trainer\,++\{++tc.resnet_cifar\,++dict\(train_step\=tc.SupervisedTrainMultiStep\(8\)\,epoch_count\=200/8\)\}/train_eval_after_multistep/25/summary.p


# Semi-supervised VAT
python run.py train "train,train_u,test:camvid{val,train,test}" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_camvid,tc.semisupervised_vat,epoch_count=20" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r
python run.py train "train,train_u,test:Cifar10{trainval,test}:(*uniform_labels(d[0]).split(index=4000),d[1])" id "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_camvid,tc.semisupervised_vat,epoch_count=20" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r

# rsync
rsync -avzhe ssh --progress pretrained_parameters/ igrubisic@treebeard:/home/igrubisic/data/pretrained_parameters/


# Semi-supervised consistency

python run.py train "train,train_u,test:Cifar10{trainval,test}:(rotating_labels(d[0])[:4000],d[0],d[1])" id "WRN,backbone_f=t(depth=28,width_factor=2,small_input=True)" "tc.wrn_cifar,tc.semisup_cons_phtps20,batch_size=[128,512],eval_batch_size=640,epoch_count=1000,train_step=ts.SemisupVATTrainStep(consistency_loss_on_labeled=False)"

python run.py train "train,train_u,test:Cityscapes{train,val}:(*d[0].split(1/8),d[1])" "standardize(mean=[73.15/255,82.9/255,72.3/255],std=[47.67/255,48.49/255,47.73/255])" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes,tc.semisup_cons_phtps20_seg,batch_size=12,eval_batch_size=12,epoch_count=200" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"