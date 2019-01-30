import argparse

import torch

from _context import vidlu
from vidlu.data import DatasetFactory
from vidlu.parsing_factories import parse_datasets, parse_model

import dirs

# Argument parsing #################################################################################

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('data', type=str, help=parse_datasets.help)
parser.add_argument('model', type=str,
                    help='Model configuration, e.g. "resnet(18, head=\'semseg\')", "wrn(28,10)".')
parser.add_argument('training', type=str,
                    help='Model configuration, e.g. "discriminative(epochs=100, )", "wrn(28,10)".')
parser.add_argument('problem', type=str, default="auto",
                    help='Problem configuration, e.g. "auto", "uncertainty".')

args = parser.parse_args()
print(args)

# data
datasets = parse_datasets(args.data, DatasetFactory(dirs.DATASETS))
print('Data:', datasets)

# model
model_name, model_f = parse_model(args.model)
model = model_f()
del model_f
model(torch.Tensor([x for x in datasets[0][:5]]))
print('Model:', model_name, model)

model.initialize_parameters()
training_config = args.training
print(training_config)

# python .\run.py "whitenoise{trainval,test}" "ResNet,backbone_f=t(depth=34,small_input=True),head_f=partial(vidlu.nn.com.classification_head,10)" ResNetCifarTrainer "accuracy"

"""
* oblikovanje komponenti za algoritme učenja i njihovo parsiranje
* inicijalizacija Imagenetom
* spremanje / učitavanje modela za resnet i densenet
* evaluacijske metrike
* spremannje informacija o učenju
* testirati, popraviti pogreške

* Reproducirati CIFAR za Densenet i ResNet

* Reproducirati Ladder-Densenet na Cityscapesu

* Reproducirati jedan članak s prepoznavanjem anomalija autoenkoderom

* VAT, polunadzirana adaptacija domene 
* Lee, GAN
* Petrin sustav za preudolabeling
"""