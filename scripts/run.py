import argparse
from pathlib import Path

#import cuda_device_order_pci_bus_id
import torch

from _context import vidlu
from vidlu.parsing_factories import parse_datasets, parse_model, parse_trainer, parse_metrics
from vidlu.utils.indent_print import indent_print
from vidlu.utils.path import to_valid_path
from vidlu import gpu_utils

from ignite import handlers

import dirs

"""
python run.py
"whitenoise{train,test}" "ResNet,backbone_f=t(depth=34,small_input=True),head_f=partial(vidlu.nn.components.classification_head,10)" ResNetCifarTrainer "accuracy"
"Cifar10{train,val}" "ResNet,backbone_f=t(depth=18,small_input=True)" ResNetCifarTrainer ""
"Cifar10{train,val}" "ResNet,backbone_f=t(depth=10,small_input=True,base_width=4)" ResNetCifarTrainer ""
"""

# Arguments ########################################################################################

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('data', type=str, help=parse_datasets.help)
parser.add_argument('model', type=str, help=parse_model.help)
parser.add_argument('trainer', type=str, help=parse_trainer.help)
parser.add_argument('evaluation', type=str, default="",
                    help='A comma-separated list of metrics.')
parser.add_argument("-d", "--device", type=torch.device, help="device", default=None)
parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1)

args = parser.parse_args()

with indent_print("Arguments:"):
    print(args)

# Device ###########################################################################################

device = args.device
if device is None:
    with indent_print("Selecting device..."):
        device = gpu_utils.get_device()

# Data #############################################################################################

with indent_print('Initializing data...'):
    datasets = parse_datasets(args.data, dirs.DATASETS, dirs.CACHE)

# Model ############################################################################################

with indent_print('Initializing model...'):
    model = parse_model(args.model, dataset=datasets[0], device=args.device,
                        verbosity=args.verbosity)

# Trainer ##########################################################################################

with indent_print('Initializing trainer...'):
    trainer = parse_trainer(args.trainer, model=model, dataset=datasets[0], device=args.device,
                            verbosity=args.verbosity)

# Evaluation #######################################################################################

with indent_print('Initializing evaluation metrics...'):
    metric_fs = parse_metrics(args.evaluation, dataset=datasets[0])

    for m in metric_fs:
        trainer.attach_metric(m())

# Learner saving ###################################################################################

learner_name = f"{to_valid_path(args.model)}-{to_valid_path(args.trainer)}"
print('Learner name:', learner_name)
learner_path = dirs.SAVED_STATES / Path(to_valid_path(args.data)) / learner_name

# Training loop ####################################################################################

print('Starting training:...')
trainer.training.iteration_completed.add_handler(lambda t: print(t.state.output.loss))
trainer.train(*datasets)

"""
def wrap_call(call):
    def wrapper(self, *a, **k):
        if isinstance(self, ScopedModule):
            print(self.scope)
        else:
            print(type(self))
        call(self, *a, **k)

    return wrapper


nn.Module.__call__ = wrap_call(nn.Module.__call__)
"""

"""from torch.utils.data import DataLoader

dl = DataLoader(datasets[0].map(lambda r: tuple(r)), num_workers=1)

for i, batch in enumerate(dl):
    print(i, batch[0].shape)"""
