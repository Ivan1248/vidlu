import argparse

import torch

from _context import vidlu

from vidlu.data import DatasetFactory
from vidlu.parsing_factories import parse_datasets, parse_model

import dirs

# Argument parsing #################################################################################

from vidlu.nn.modules import Sequential
from vidlu import models


model_name, model_f = parse_model(
    "ResNet,backbone_f=ArgTree(depth=18,small_input=True),head_f=partial(components.classification_head,10)")
    #"DenseNet,backbone_f=t(depth=121,small_input=True),head_f=partial(vidlu.nn.components.classification_head,10)")

model = model_f()
output = model(torch.randn(8, 3, 32, 32))
model.initialize()
print(output.shape)
print(output)
# python .\run.py "whitenoise{trainval,test}" "ResNet,backbone_f=t(depth=34,small_input=True),head_f=partial(vidlu.nn.components.classification_head,10)" "ResNetCifar" "accuracy"
