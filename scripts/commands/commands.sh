# Symbol resolution in these strings:
#  - trainer configs are bare (resnet_cifar, adversarial, ...): get_trainer injects vidlu.configs.training;
#  - dataset/model classes are bare: get_data/get_model inject them;
#  - other vidlu modules use full-word handles from STANDARD_PRE: modules, components, initialization,
#    steps, attacks, jitter, transforms, configs, training, optim, schedulers; plus partial, t (ArgTree).
#  Extend/override per run with --pre "<python statements>" / --imports "alias=module,...".
#  Extensions are available prefixed, e.g. irap_gaim.<name> (see the IRAP section below).
python run.py train "dict(train=MNIST('trainval')[:54000], val=MNIST('trainval')[54000:])" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar"
# Smoke test on a tiny labeled dummy dataset (3 classes; class_count/problem come from its info).
python run.py train "(DummyClassification(), DummyClassification())" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar" --metrics "accuracy"
# Smoke test on tiny labeled blobs (3 classes; blurred random blobs colored per class).
python run.py train "(ColoredBlobs(), ColoredBlobs())" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar" --metrics "accuracy"
python run.py train "dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])" id "SmallImageClassifier" "SmallImageClassifierTrainer,data_loader_f=t(num_workers=4)" "" -v 2
python run.py train "dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])" id "DiscriminativeModel,backbone_f=partial(components.PreactBlock,kernel_sizes=[3,3,3],base_width=64,width_factors=[1,1,1]),init=partial(initialization.kaiming_resnet)" ResNetCifarTrainer ""
python run.py train "dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])" standardize "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar"
python run.py train "dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar"
python run.py train "dict(train=Cifar10('trainval')[:45000], val=Cifar10('trainval')[45000:])" id "DenseNet,backbone_f=t(depth=121,small_input=True)" "resnet_cifar"
# Shorter data:     "dict(zip(['train', 'val'], Cifar10('trainval').split(45000)))"

# CIFAR
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV1,backbone_f=t(depth=18,small_input=True)" "resnet_cifar"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "WideResNet,backbone_f=t(depth=28,width_factor=10,small_input=True)" "resnet_cifar"
# Shorter data:     "Cifar10('trainval'), Cifar10('test')"
# Shorter data:     "map(Cifar10, ['trainval', 'test'])"

python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "DenseNet,backbone_f=t(depth=121,small_input=True)" "densenet_cifar"
## i-RevNet
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(4,)*4,block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "irevnet_cifar"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(2,6,6,2),block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "irevnet_cifar"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id  "IRevNet,backbone_f=t(init_stride=1,base_width=12,group_lengths=(2,6,6,2),block_f=t(kernel_sizes=(3,3),width_factors=(1,1)))" "irevnet_hybrid_cifar"
python run.py train "(Cifar10('trainval')[:100], Cifar10('test')[:100])" id "ResNetV2,backbone_f=t(depth=10,small_input=True)" "resnet_cifar"

# Multistep step
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True)"  "resnet_cifar,train_step=SupervisedTrainMultiStep(8),epoch_count=200/8"

# semantic segmentation

## DenseNet (pretrained on ImageNet)
python run.py train "dict(train=Cityscapes('train', downsampling=2), val=Cityscapes('val', downsampling=2))" id "DenseNet,backbone_f=t(depth=121)" "ladder_densenet,optimizer_maker=FineTuningOptimizerMaker({'backbone':1/5})" --params "densenet(backbone),backbone:densenet121-a639ec97.pth"


## ResNet
python run.py train "dict(train=CamVid('train', downsampling=2), val=CamVid('val', downsampling=2))" id "ResNetV1,backbone_f=t(depth=18)" "swiftnet_cityscapes,jitter=jitter.SegRandomCropHFlip((360,480)),optimizer_maker=None" -r

## Swiftnet
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes"
python run.py train "dict(train=Cityscapes('train', downsampling=2), val=Cityscapes('val', downsampling=2))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes"
python run.py train "(CamVid('train').join(CamVid('val')), CamVid('test'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_camvid"
# network deconvolution
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" id "SwiftNet,backbone_f=t(depth=18,block_f=t(norm_f=None,conv_f=partial(modules.DeconvConv,bias=True,padding='half')))" "swiftnet_cityscapes"

## SwiftNet pretrained on Cityscapes
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(mean=[.485,.456,.406],std=[.229,.224,.225])" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes" --params swiftnet:swiftnet_ss_cs_best.pt
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes" --params swiftnet:rn18_single_scale/model.pt  # 75.39
## SwiftNet pretrained on ImageNet
python run.py train "(CamVid('train').join(CamVid('val')), CamVid('test'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_camvid" --params "resnet[backbone]->backbone.backbone:resnet18"
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes" --params "resnet[backbone]->backbone.backbone:resnet18"
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes" --params "resnet:backbone->backbone.backbone:resnet18"  # 75.39

### SwiftNet-RN50 (non)robust pretraining
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=50)" "swiftnet_cityscapes" --params "resnet[backbone]->backbone.backbone:resnet50"
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=50)" "swiftnet_cityscapes" --params "madrylab_resnet[backbone]->backbone.backbone:madrylab/cvrobust/resnet-50-imagenet.pt"
### bs8 ep250
CUDA_VISIBLE_DEVICES=0 python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes,epoch_count=250,batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18"


## SegResNetV1 pretrained
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" standardize "SegResNetV1,backbone_f=t(depth=50)" "ladder_densenet"

## SegResNetV1 pretrained
python run.py train "(CamVid('train').join(CamVid('val')), CamVid('test'))" standardize "SegResNetV1,backbone_f=t(depth=50)" "swiftnet_camvid,epoch_count=400,optimizer_maker=FineTuningOptimizerMaker({'backbone': 0})" --params "resnet[backbone]->backbone:resnet50"
python run.py train "(CamVid('train').join(CamVid('val')), CamVid('test'))" standardize "SegResNetV1,backbone_f=t(depth=50)" "swiftnet_camvid,epoch_count=400,optimizer_maker=FineTuningOptimizerMaker({'backbone': 0})" --params "madrylab_resnet[backbone]->backbone.backbone:madrylab/cvrobust/resnet-50-imagenet.pt"


# Adversarial training
## ResNet
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar,adversarial,attack_f=partial(madry_Cifar10_attack,step_count=7),eval_attack_f=t(step_count=10,stop_on_success=True)"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar,adversarial,train_step=AdversarialTrainStep(True),attack_f=partial(madry_Cifar10_attack,step_count=7,stop_on_success=True),eval_attack_f=t(step_count=20)"
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True)" "resnet_cifar,adversarial,attack_f=partial(madry_Cifar10_attack,step_count=7),eval_attack_f=partial(madry_Cifar10_attack,step_count=20),train_step=AdversarialTrainMultiStep(),epoch_count=25"

## Swiftnet
python run.py train "dict(train=Cityscapes('train'), val=Cityscapes('val'))" standardize "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes,epoch_count=40,batch_size=1,adversarial,train_step=AdversarialTrainStep(False),attack_f=partial(madry_Cifar10_attack, step_count=40, eps=48/255, clip_bounds=None)" --params swiftnet:swiftnet_ss_cs.pt

# Tent activations
python run.py train "(Cifar10('trainval'), Cifar10('test'))" id "ResNetV2,backbone_f=t(depth=18,small_input=True,block_f=t(act_f=components.Tent))" "wrn_cifar_tent,adversarial},attack_f=attacks.DummyAttack,eval_attack_f=partial(madry_Cifar10_attack,step_count=7,stop_on_success=True)"
python run.py train "(MNIST('trainval'), MNIST('test'))" id "MNISTNet,backbone_f=t(act_f=components.Tent,use_bn=False)" "mnistnet_tent,adversarial,attack_f=attacks.DummyAttack,eval_attack_f=mnistnet_tent_eval_attack"

# show_summary.py
python show_summary.py ~/data/states/Cifar10\{trainval\,test\}/ResNetV2\,backbone_f\=t\(depth\=18\,small_input\=True\)/Trainer\,++\{++resnet_cifar\,++dict\(train_step\=SupervisedTrainMultiStep\(8\)\,epoch_count\=200/8\)\}/train_eval_after_multistep/25/summary.p


# Semi-supervised VAT
python run.py train "dict(train=CamVid('val'), train_u=CamVid('train'), test=CamVid('test'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_camvid,semisupervised_vat,epoch_count=20" --params "resnet[backbone]->backbone.backbone:resnet18" -r
# Two equivalent forms for a single-computation labeled/unlabeled split (train_u = the unlabeled remainder after the labeled 4000):
#   zip:    "dict(zip(('train', 'train_u'), rotating_labels(Cifar10('trainval')).split(index=4000)), test=Cifar10('test'))"
#   lambda: "(lambda tv: dict(train=tv[:4000], train_u=tv[4000:], test=Cifar10('test')))(rotating_labels(Cifar10('trainval')))"
python run.py train "dict(zip(('train', 'train_u'), rotating_labels(Cifar10('trainval')).split(index=4000)), test=Cifar10('test'))" id "SwiftNet,backbone_f=t(depth=18)" "swiftnet_camvid,semisupervised_vat,epoch_count=20" --params "resnet[backbone]->backbone.backbone:resnet18" -r

# rsync
rsync -avzhe ssh --progress pretrained/ igrubisic@treebeard:/home/igrubisic/data/pretrained/


# Semi-supervised consistency (old)

#  CIFAR
python run.py train "dict(train=rotating_labels(Cifar10('trainval'))[:4000], train_u=Cifar10('trainval'), test=Cifar10('test'))" id "WRN,backbone_f=t(depth=28,width_factor=2,small_input=True)" "wrn_cifar,semisup_cons_phtps20,batch_size=[128,512],eval_batch_size=640,epoch_count=1000,train_step=steps.SemisupVATTrainStep(consistency_loss_on_labeled=False)"

# CS
python run.py train "dict(train=Cityscapes('train'), train_u=Cityscapes('train'), test=Cityscapes('val'))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes,semisup_cons_phtps20_seg,batch_size=8,eval_batch_size=4,epoch_count=300,batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

# CS halfres
python run.py train "dict(train=Cityscapes('train', downsampling=2), train_u=Cityscapes('train', downsampling=2), test=Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,semisup_cons_phtps20_seg,epoch_count=600" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

python run.py train "dict(train=Cityscapes('train', downsampling=2), val=Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,epoch_count=300" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

# supervised
CUDA_VISIBLE_DEVICES=0 python run.py train "dict(train=Cityscapes('train', downsampling=2), val=Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,epoch_count=300" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

CUDA_VISIBLE_DEVICES=0 python run.py train "(Cityscapes('train', downsampling=2).permute(53)[:1448], Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,epoch_count=300*2" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

CUDA_VISIBLE_DEVICES=0 python run.py train "(Cityscapes('train', downsampling=2).permute(53)[:744], Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,epoch_count=300*4" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"

CUDA_VISIBLE_DEVICES=0 python run.py train "(Cityscapes('train', downsampling=2).permute(53)[:372], Cityscapes('val', downsampling=2))" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18)" "swiftnet_cityscapes_halfres,epoch_count=300*8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth"
# Shorter data string:                     "(lambda f=partial(Cityscapes, downsampling=2): (f('train').permute(53)[:372], f('val')))()"
# Shorter data string:                     "(lambda f: (f('train').permute(53)[:372], f('val')))(partial(Cityscapes, downsampling=2))"
# Shorter data string:                     "((f:=partial(Cityscapes, downsampling=2))('train').permute(53)[:372], f('val'))"


# https://docs.google.com/spreadsheets/d/1Tqydw1Hhvjyflo412UJJ1Y6720sRpCkYUjH3FRvYeLw/edit#gid=851257437

# IRAP
# Jitter is configured in irap_gaim.irap_local_rec_trainer (dataset transforms remain deterministic).

IRAP_HOME=~/projects/irap_home CUDA_VISIBLE_DEVICES=1 python -m pdb scripts/run.py train "irap_gaim.make_bih_data()" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])" -e 2 -r restart

# IRAP context comparison: [-4,-1,0] vs [-1,0] vs [0]
# context_offsets=(0,-1,-4) is the default; (0,-1) drops -40min; (0,) is current frame only.
# sequence_length must match len(context_offsets). Metrics use matching context for consistent class distribution.

## 3-frame baseline [-4, -1, 0] (default context_offsets)
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_bih_data()" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=3,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data()['train'])"

## 2-frame [-1, 0]
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_bih_data(context_offsets=(0,-1))" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=2,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data(context_offsets=(0,-1))['train'])"

## 1-frame [0]
CUDA_VISIBLE_DEVICES=0 IRAP_HOME=~/projects/irap_home python scripts/run.py train "irap_gaim.make_bih_data(context_offsets=(0,))" "standardize" "irap_gaim.ImageSequenceClassifier,class_counts=irap_gaim.get_class_counts(),attention=False,sequence_length=1,encoder_f=partial(irap_gaim.ResNetEncoder,pretrained=False)" "irap_gaim.irap_local_rec_trainer" --params "id[backbone]->frame_encoder.resnet:irap_gaim/vistas.pt" --metrics "irap_gaim.get_irap_metrics(irap_gaim.make_bih_data(context_offsets=(0,))['train'])"