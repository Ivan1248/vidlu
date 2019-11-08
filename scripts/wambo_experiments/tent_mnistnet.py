from functools import partial
import tempfile
from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from _context import vidlu
from vidlu.data.datasets import MNIST
from vidlu import models, factories, experiments
from vidlu.modules import components
from vidlu.modules.components import Tent
from vidlu.modules.other.mnistnet import MNISTNetBackbone
from vidlu.training import Trainer, configs, adversarial, initialization, metrics, extensions
from vidlu.utils import logger
import vidlu.training.steps as ts

# Data

data_dir = Path(tempfile.gettempdir()) / 'datasets'
data_dir.mkdir(exist_ok=True)
data_dir = data_dir / 'mnist'
data = dict(train=MNIST(data_dir, 'trainval'), test=MNIST(data_dir, 'test'))
prepare = factories.get_data_preparation(data['train'])  # convert to torch and [0, 1] range
data = dict(**{k: prepare(v) for k, v in data.items()})


# Model

class MNISTNetTentModel(models.SeqModel):
    def __init__(self):
        super().__init__(
            seq=dict(
                backbone=MNISTNetBackbone(
                    act_f=partial(Tent, channelwise=False, delta_range=(0.05, 1.)),
                    use_bn=True),
                head=components.heads.ClassificationHead1D(10)),
            init=initialization.kaiming_mnistnet)


model = MNISTNetTentModel()
device = torch.device('cuda:0')
model.initialize(next(iter(DataLoader(data['train'], batch_size=2)))[0])
model.to(device)


# Trainer

def create_optimizer(trainer):
    delta_params = [v for k, v in model.named_parameters() if k.endswith('delta')]
    other_params = [v for k, v in model.named_parameters() if not k.endswith('delta')]
    return optim.Adam([dict(params=other_params), dict(params=delta_params, weight_decay=0.12)],
                      lr=1e-3, weight_decay=0)


trainer = Trainer(
    model=model,
    extend_output=configs.classification_extend_output,
    loss_f=nn.CrossEntropyLoss,
    train_step=configs.AdversarialTrainStep(),
    # no attack is used during training (the training is not adversarial)
    eval_step=ts.AdversarialEvalStep(),
    epoch_count=40,
    batch_size=100,
    optimizer_maker=create_optimizer,
    extensions=extensions.AdversarialTraining(
        attack_f=adversarial.attacks.DummyAttack,
        eval_attack_f=partial(adversarial.attacks.PGDAttack, eps=0.3, step_size=0.1, step_count=20,
                              stop_on_success=True),
    ))

for m in [metrics.FuncAverageMetric(lambda iter_output: iter_output.loss, name='loss'),
          metrics.ClassificationMetrics(10, metrics=['A']),
          metrics.with_suffix(metrics.ClassificationMetrics, 'adv')(
              10, hard_prediction_name="other_outputs_adv.hard_prediction", metrics=['A'])]:
    trainer.add_metric(m)

# Reporting and evaluation during training

logger = logger.Logger()
experiments.define_training_loop_actions(trainer, data, logger)

# Training and evaluation

logger.log("Evaluating initially...")
trainer.eval(data['test'])
logger.log("Starting training...")
trainer.train(data['train'])
logger.log("Evaluating on training data...")
trainer.eval(data['train'])
