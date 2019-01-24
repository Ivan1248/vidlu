import argparse
import datetime

import numpy as np
from tqdm import tqdm

from _context import dl_uncertainty

from dl_uncertainty.data import DataLoader
from dl_uncertainty import dirs, training
from dl_uncertainty import data_utils, model_utils
from dl_uncertainty.utils.visualization import view_predictions, view_predictions_2
"""
Use "--trainval" only for training on "trainval" and testing "test".
CUDA_VISIBLE_DEVICES=0 python test.py
CUDA_VISIBLE_DEVICES=1 python test.py
CUDA_VISIBLE_DEVICES=2 python test.py
 --mcdropout --trainval --uncertainties
 cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-e200/2018-05-28-0956/Model
 cifar wrn 28 10 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/wrn-28-10-dropout-e200/2018-05-29-1708/Model --mcdropout
 cifar dn 100 12 /home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-28-0121/Model
 cifar rn 34 8
 cityscapes dn 121 32  /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-18-0010/Model
 cityscapes rn 50 64
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-pretrained-e30/2018-05-18-2129/Model
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-pretrained-e80/2018-05-24-0403/Model --mcdropout --test_dataset wilddash
  
 dropout0.1
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-pretrained-e30/2018-06-24-1051/Model --mcdropout
 dropout0.1 1/2
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac2-pretrained-e30/2018-06-22-1256/Model --mcdropout --trainval --uncertainties
 dropout0.1 1/4
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac4-pretrained-e30/2018-06-22-1226/Model --mcdropout --trainval --uncertainties
 dropout0.1 1/8
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac8-pretrained-e30/2018-06-22-1205/Model --mcdropout --trainval --uncertainties
 
 dropout0.2
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-pretrained-e30/2018-06-24-1941/Model --mcdropout --trainval --uncertainties
 dropout0.2 1/2
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac2-pretrained-e30/2018-06-24-1509/Model --mcdropout --trainval --uncertainties
 dropout0.2 1/4
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac4-pretrained-e30/2018-06-25-1253/Model --mcdropout --trainval --uncertainties
 dropout0.2 1/8
 camvid ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/camvid-trainval/ldn-121-32-dropout-frac8-pretrained-e30/2018-06-25-1232/Model --mcdropout --trainval --uncertainties

 dropout0.2
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-randomcrop-pretrained-e80/2018-06-27-0739/Model --mcdropout --trainval --uncertainties

  ood
 cityscapes ldn 121 32 /home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/ldn-121-32-dropout-randomcrop-pretrained-e120/2018-07-24-0416/Model --mcdropout --trainval --uncertaintyood


 mozgalo rn 50 64
 mozgalo rn 18 64
"""

parser = argparse.ArgumentParser()
parser.add_argument('ds', type=str)
parser.add_argument('net', type=str)
parser.add_argument('depth', type=int)
parser.add_argument('width', type=int)
parser.add_argument('saved_path', type=str)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--mcdropout', action='store_true')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--test_on_training_set', action='store_true')
parser.add_argument('--test_dataset', default="", type=str)
parser.add_argument('--view', action='store_true')
parser.add_argument('--view2', action='store_true')
parser.add_argument('--hard', action='store_true')  # display hard exampels
parser.add_argument('--uncertainties', action='store_true')
parser.add_argument('--uncertaintyood', action='store_true')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()
print(args)

assert not args.hard or args.hard and args.view

# Cached dataset with normalized inputs

print("Setting up data loading...")
ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
    args.ds, trainval_test=args.trainval)

model_ds = ds_train

if args.test_dataset != "":
    ds_train, ds_test = data_utils.get_cached_dataset_with_normalized_inputs(
        args.test_dataset, trainval_test=args.trainval)

# Model

print("Initializing model and loading state...")
model = model_utils.get_model(
    net_name=args.net,
    ds_train=model_ds,
    depth=args.depth,
    width=args.width,  # width factor for WRN, base_width for others
    epoch_count=1,
    dropout=args.dropout or args.mcdropout)
model.load_state(args.saved_path)

if args.view or args.hard or args.view2:
    ds_disp = ds_train if args.test_on_training_set else ds_test

    if args.hard:
        ds_disp = training.get_hard_examples(model, ds_disp)

    def predict(x):
        out_names = ['output', 'probs', 'probs_entropy']
        output, probs, probs_entropy = model.predict(
            x, single_input=True, outputs=out_names)
        if args.mcdropout:
            mc_output, mc_probs, mc_probs_entropy, mc_probs_mi = model.predict(
                x,
                single_input=True,
                mc_dropout=args.mcdropout,
                outputs=out_names + ['probs_mi']  #,'pred_logits_var']
            )
            maxent = np.log(mc_probs.shape[-1])
            epistemic = mc_probs_mi
            aleatoric = mc_probs_entropy - mc_probs_mi
            uncertainties = [
                probs_entropy, mc_probs_entropy, aleatoric,
                np.clip(epistemic * 4, 0, maxent)
            ]
            for a in uncertainties:
                a.flat[0] = maxent
                a.flat[1] = 0
            return [output, mc_output] + uncertainties
        return [output, probs_entropy]

    save_dir = f'{dirs.CACHE}/viewer/{args.ds}-{args.net}-{args.depth}-{args.width}'
    if args.test_dataset != "":
        save_dir += f'-{args.test_dataset}'
    if args.dropout or args.mcdropout:
        save_dir += "-dropout"
    if args.trainval:
        save_dir += "-test"
    view = view_predictions_2 if args.view2 else view_predictions
    view(ds_disp, predict, save_dir=save_dir if args.save else None)
elif args.uncertainties:
    aleatoric = []
    epistemic = []
    for x, y in tqdm(ds_test):
        ent, mi = model.predict(
            x,
            single_input=True,
            mc_dropout=True,
            outputs=['probs_entropy', 'probs_mi'])
        aleatoric += [np.mean(ent - mi)]
        epistemic += [np.mean(mi)]
    aleatoric = np.mean(aleatoric)
    epistemic = np.mean(epistemic)
    print(aleatoric, epistemic)
elif args.uncertaintyood:
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    pixelwise = True
    y_true = np.ones(156)
    y_true[141:] = 0
    y_true[[141, 142, 151, 155]] = -1
    ds_test = ds_test.subset([i for i, v in enumerate(y_true) if v != -1])
    y_true = np.ones(156 - 4)
    y_true[141:] = 0
    epistemic = []
    for x, y in tqdm(ds_test):
        ent, mi = model.predict(
            x,
            single_input=True,
            mc_dropout=True,
            outputs=['probs_entropy', 'probs_mi'])
        #mi = ent
        if pixelwise:
            epistemic += [mi]
        else:
            epistemic += [np.mean(mi)]
        print(epistemic[-1])

    y_score = np.array(epistemic).flatten()
    if pixelwise:
        y_true = np.repeat(y_true, len(epistemic[0].flat))

    #p_in, r_in, thr = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    #pi_in = np.array([np.max(p_in[:j + 1]) for j, _ in enumerate(p_in)])

    #p_out, r_out, thr = precision_recall_curve(
    #    y_true=1 - y_true, probas_pred=-y_score)
    #pi_out = np.copy(p_out)
    #pi_out = np.array([np.max(p_out[:j + 1]) for j, _ in enumerate(p_out)])

    #ap_in = auc(r_in, p_in)
    #api_in = auc(r_in, pi_in)
    #ap_out = auc(r_out, p_out)
    #api_out = auc(r_out, pi_out)
    #print(ap_in, api_in, ap_out, api_out)
    ap_in = average_precision_score(y_true, -y_score)
    ap_out = average_precision_score(1 - y_true, y_score)
    print(ap_in, ap_out)
else:
    model.test(
        DataLoader(ds_train, model.batch_size),
        "training set",
        mc_dropout=args.mcdropout)
    model.test(
        DataLoader(ds_test, model.batch_size),
        "test set",
        mc_dropout=args.mcdropout)
