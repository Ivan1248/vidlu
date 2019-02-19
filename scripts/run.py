import argparse
from pathlib import Path
import datetime
from functools import partial

import cuda_pci_bus_id_device_order
import torch

from _context import vidlu
from vidlu.nn.modules import parameter_count
from vidlu.factories import parse_datasets, parse_model, parse_trainer, parse_metrics
from vidlu.utils.indent_print import indent_print
from vidlu.utils.path import to_valid_path
from vidlu import gpu_utils

from ignite import handlers

import dirs

"""
python run.py
"whitenoise{train,test}" "ResNet,backbone_f=t(depth=34,small_input=True),head_f=partial(vidlu.nn.components.classification_head,10)" ResNetCifarTrainer "accuracy"
"Cifar10{train,val}" "ResNet,backbone_f=t(depth=18,small_input=True)" ResNetCifarTrainer ""
"Cifar10{train,val}" "ResNet,backbone_f=t(depth=18,small_input=True),init=t(zero_init_residual=False)" ResNetCifarTrainer ""
"Cifar10{train,val}" "ResNet,backbone_f=t(depth=10,small_input=True,base_width=4)" ResNetCifarTrainer ""
"Cifar10{train,val}" "DiscriminativeModel,backbone_f=c.VGGBackbone,init=partial(init.kaiming_resnet)" ResNetCifarTrainer ""
"Cifar10{train,val}" "SmallImageClassifier" "SmallImageClassifierTrainer,data_loader_f=t(num_workers=4)" "" -v 2
"Cifar10{train,val}" "DiscriminativeModel,backbone_f=partial(c.PreactBlock,kernel_sizes=[3,3,3],base_width=64,width_factors=[1,1,1]),init=partial(init.kaiming_resnet)" ResNetCifarTrainer ""
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
        device = torch.device(gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))

# Data #############################################################################################

with indent_print('Initializing data...'):
    datasets = parse_datasets(args.data, dirs.DATASETS, dirs.CACHE)

# Model ############################################################################################

with indent_print('Initializing model...'):
    model = parse_model(args.model, dataset=datasets[0], device=args.device,
                        verbosity=args.verbosity)
    model.to(device=device)
print(model)
print('Parameter count:', parameter_count(model))

# Trainer ##########################################################################################

with indent_print('Initializing trainer...'):
    trainer = parse_trainer(args.trainer, model=model, dataset=datasets[0], device=args.device,
                            verbosity=args.verbosity)

# Evaluation #######################################################################################


with indent_print('Initializing evaluation metrics...'):
    metrics = [m() for m in parse_metrics(args.evaluation, dataset=datasets[0])]
    for m in metrics:
        trainer.attach_metric(m)

# Learner saving ###################################################################################

learner_name = f"{to_valid_path(args.model)}-{to_valid_path(args.trainer)}"
print('Learner name:', learner_name)
learner_path = dirs.SAVED_STATES / Path(to_valid_path(args.data)) / learner_name


# Reporting and logging ############################################################################

def eval_str(metrics):
    return ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])


def log(text: str):
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    text = f"[{time_str}] {text}"
    print(text)


# Training loop ####################################################################################

print('Starting training:...')


@trainer.training.epoch_started.add_handler
def on_training_epoch_started(e):
    s = e.state
    trainer.eval(datasets[1])
    log(f"Starting epoch {s.epoch}/{s.max_epochs} ({s.batch_count} batches)")


@trainer.training.completed.add_handler
def on_training_completed(e):
    trainer.eval(datasets[1])


def report_metrics(e, validation=False):
    s = e.state
    metrics = trainer.get_metrics(reset=True)
    with indent_print():
        epoch_fmt, iter_fmt = f'{len(str(s.max_epochs))}d', f'{len(str(s.batch_count))}d'
        log(f'{format(s.epoch, epoch_fmt)}.'
            + ('val/test' if validation
               else f'{format((s.iteration - 1) % s.batch_count + 1, iter_fmt)}')
            + ': ' + eval_str(metrics))


trainer.evaluation.epoch_completed.add_handler(partial(report_metrics, validation=True))


@trainer.training.iteration_completed.add_handler
def on_iteration_completed(e):
    if e.state.iteration % (e.state.batch_count // 5) == 0:
        report_metrics(e)


trainer.train(datasets[0].cache())

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
