import argparse
from datetime import datetime
from functools import partial

import set_cuda_order_pci  # CUDA_DEVICE_ORDER = "PCI_BUS_ID"
import torch
import numpy as np

from _context import vidlu
from vidlu.modules.elements import parameter_count
from vidlu.factories import get_data, get_model, get_trainer, get_metrics
from vidlu.training.checkpoint_manager import CheckpointManager
from vidlu.utils.indent_print import indent_print
from vidlu.utils.path import to_valid_path
from vidlu import gpu_utils, defaults

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

# Argument parsing #################################################################################

parser = argparse.ArgumentParser(description='Training script')
# learner configuration
parser.add_argument('data', type=str, help=get_data.help)
parser.add_argument('model', type=str, help=get_model.help)
parser.add_argument('trainer', type=str, help=get_trainer.help)
parser.add_argument('evaluation', type=str, default="", help='A comma-separated list of metrics.')
# device
parser.add_argument("-d", "--device", type=torch.device, help="PyTorch device.", default=None)
# experiment result saving, state checkpoints
parser.add_argument("-e", "--experiment_suffix", type=str, default='',
                    help="Experiment ID suffix. Required for running multiple experiments with the"
                         + " same configuration.")
parser.add_argument("-r", "--resume", action='store_true',
                    help="Resume training from the last checkpoint of the same experiment.")
parser.add_argument("--restart", action='store_true',
                    help="Remove existing checkpoints and start training from the beginning.")
parser.add_argument("-s", "--seed", type=str, default=53,
                    help="RNG seed for experiment reproducibility. WARNING: CUDA not supported.")
# reporting
parser.add_argument("-v", "--verbosity", type=int, help="Console output verbosity.", default=1)

args = parser.parse_args()

with indent_print("Arguments:"):
    print(args)

# RNG seed #########################################################################################

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Device ###########################################################################################

device = args.device
if device is None:
    with indent_print("Selecting device..."):
        device = torch.device(
            gpu_utils.get_first_available_device(max_gpu_util=0.5, no_processes=False))

# Data #############################################################################################

with indent_print('Initializing data...'):
    ds_train, ds_test = get_data(args.data, dirs.DATASETS, dirs.CACHE)

ds_train_jittered = ds_train.map(defaults.get_jitter(ds_train))
prepare_input = defaults.get_input_preparation(ds_train)
ds_train, ds_train_jittered, ds_test = [ds.map(prepare_input)
                                        for ds in [ds_train, ds_train_jittered, ds_test]]

# Model ############################################################################################

with indent_print('Initializing model...'):
    model = get_model(args.model, dataset=ds_train, device=args.device, verbosity=args.verbosity)
    model.to(device=device)
print(model)
print('Parameter count:', parameter_count(model))

# Logging ##########################################################################################

log_lines = []


def log(text: str):
    time_str = datetime.now().strftime('%H:%M:%S')
    text = f"[{time_str}] {text}"
    log_lines.append(text)
    print(text)


# Trainer and evaluation metrics ###################################################################

with indent_print('Initializing trainer...'):
    trainer = get_trainer(args.trainer, model=model, dataset=ds_train, device=args.device,
                          verbosity=args.verbosity)

with indent_print('Initializing evaluation metrics...'):
    metrics = [m() for m in get_metrics(args.evaluation, dataset=ds_train)]
    for m in metrics:
        trainer.attach_metric(m)

# Checkpoint manager ###############################################################################

learner_name = to_valid_path(f"{args.model}-{args.trainer}")
experiment_id = f'{args.data}-{learner_name}-exp_{args.experiment_suffix}'
print('Learner name:', learner_name)
print('Experiment ID:', experiment_id)
cpman = CheckpointManager(dirs.SAVED_STATES, experiment_str=experiment_id, resume=args.resume,
                          remove_old=args.experiment_suffix == '' and not args.resume)
if args.resume:
    state, log_lines = cpman.load_last()
    for line in log_lines:
        print(line)
    trainer.load_state_dict(state)
    model.to(device=device)


# Training loop actions ############################################################################

# trainer.evaluation.iteration_completed.add_handler(ProgressBar()._update)

@trainer.training.epoch_started.add_handler
def on_epoch_started(en):
    s = en.state
    log(f"Starting epoch {s.epoch}/{s.max_epochs}"
        + f" ({s.batch_count} batches, lr={trainer.lr_scheduler.get_lr()})")


@trainer.training.epoch_completed.add_handler
def on_epoch_completed(engine=None):
    trainer.eval(ds_test)
    cpman.save(trainer.state_dict(), summary=log_lines)  # checkpoint


def report_metrics(en, validation=False):
    def eval_str(metrics):
        return ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])

    s = en.state
    metrics = trainer.get_metric_values(reset=True)
    with indent_print():
        epoch_fmt, iter_fmt = f'{len(str(s.max_epochs))}d', f'{len(str(s.batch_count))}d'
        iter = en.state.iteration % en.state.batch_count
        log(('val' if validation
             else f'{format(s.epoch, epoch_fmt)}.'
                  + f'{format((iter - 1) % s.batch_count + 1, iter_fmt)}')
            + ': ' + eval_str(metrics))


@trainer.training.iteration_completed.add_handler
def on_iteration_completed(en):
    if en.state.iteration % en.state.batch_count % (en.state.batch_count // 5) == 0:
        report_metrics(en)


trainer.evaluation.epoch_completed.add_handler(
    partial(report_metrics, validation=True))

# Training #########################################################################################


print(('Continuing' if args.resume else 'Starting') + ' training:...')

trainer.eval(ds_test)
trainer.train(ds_train_jittered, restart=False)

cpman.remove_old_checkpoints()

print(f'Trained model saved in {cpman.exp_dir_path}')

# End ##############################################################################################

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
