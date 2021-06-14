set -e

echo "Reproducing figure with outputs of models trained on half-resolution Cityscapes train with 1/4 of the labels"

echo "Generaiing inputs..."
CUDA_VISIBLE_DEVICES=0 python run.py test "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(53),4)[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r --module "papers.semisup_efficient_mva21.generate_images:generate_inputs,e,n=8,dir='/tmp/semisup'"

echo "Generaiing results (semi-supervised MT-PhTPS)..."
CUDA_VISIBLE_DEVICES=0 python run.py test "train,train_u,test:Cityscapes(downsampling=2){train,val}:(folds(d[0].permute(53),4)[0],d[0],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,tc.semisup_cons_phtps20_seg,train_step=ts.MeanTeacherStep(alpha=0.5),lr_scheduler_f=lr.QuarterCosLR,epoch_count=800,batch_size=[8,8]" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r --module "papers.semisup_efficient_mva21.generate_images:generate_results,e,dir='/tmp/semisup'"
echo "Generaiing results (supervised)..."
CUDA_VISIBLE_DEVICES=0 python run.py test "train,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(53)[:744],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=partial(lr.LambdaLR, lr_lambda=lambda e:math.cos(e/800*math.pi/2)),epoch_count=800,batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r --module "papers.semisup_efficient_mva21.generate_images:generate_results,e,dir='/tmp/semisup',suffix='sup'"

echo "Producing LaTeX code..."
CUDA_VISIBLE_DEVICES=0 python run.py test "train,test:Cityscapes(downsampling=2){train,val}:(d[0].permute(53)[:744],d[1])" "standardize(cityscapes_mo)" "SwiftNet,backbone_f=t(depth=18,small_input=False)" "tc.swiftnet_cityscapes_halfres,lr_scheduler_f=partial(lr.LambdaLR, lr_lambda=lambda e:math.cos(e/800*math.pi/2)),epoch_count=800,batch_size=8" --params "resnet[backbone]->backbone.backbone:resnet18-5c106cde.pth" -r --module "papers.semisup_efficient_mva21.generate_images:latex_grid,n=8,dir='/tmp/semisup'"
