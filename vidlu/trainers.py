from functools import partial, partialmethod, lru_cache

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from ignite import engine
from ignite.engine import Engine, Events

from vidlu.data import DataLoader
from vidlu.utils.misc import Event
from vidlu.utils.func import argtree_partialmethod, ArgTree, default_args
from vidlu.utils.collections import NameDict
from vidlu.utils.torch import prepare_batch
from vidlu.metrics import FuncMetric
from vidlu.engine import Engine


# Runner ###########################################################################################


class Evaluator:
    def __init__(self, model, data_loader_f=partial(DataLoader, batch_size=1, num_workers=2),
                 metrics={}, prepare_batch=prepare_batch, device=None):
        super().__init__()
        self.model = model
        self.prepare_batch = partial(prepare_batch, device=device, non_blocking=False)
        self.data_loader_f = data_loader_f

        self.evaluation = Engine(lambda e, b: self.eval_batch(b))

    def __getattr__(self, item):
        return getattr(self.evaluation, item)

    def _attach_metric(self, metric, name=None, engine=None):
        name = name or type(metric).__name__
        engine = engine or self.evaluation
        engine.epoch_started.add_handler(lambda e: metric.reset())
        engine.iteration_completed.add_handler(lambda e: metric.update(e.state.output))

        def compute_metric(e):
            e.state.metrics[name] = metric.compute()

        engine.epoch_completed.add_handler(lambda e: compute_metric(e))

    def attach_metric(self, metric, name=None):
        self._attach_metric(metric, name=name, engine=self.evaluation)

    def eval(self, dataset):
        return self.evaluation.run(self.data_loader_f(dataset))

    def eval_batch(self, batch):
        raise NotImplementedError()


# Trainer ##########################################################################################

class Trainer(Evaluator):
    def __init__(self, model, loss_f, weight_decay, optimizer_f, epoch_count,
                 lr_scheduler_f=partial(LambdaLR, lr_lambda=lambda e: 1), batch_size=1,
                 data_loader_f=default_args(Evaluator).data_loader_f, metrics={},
                 prepare_batch=prepare_batch, device=None):
        self.model = model
        self.loss = loss_f()
        self.optimizer = optimizer_f(params=self.model.parameters(), weight_decay=weight_decay)
        self.epoch_count = epoch_count
        self.lr_scheduler = lr_scheduler_f(optimizer=self.optimizer)

        super().__init__(model, data_loader_f=partial(data_loader_f, batch_size=batch_size),
                         metrics=dict(loss=FuncMetric(self.loss), **metrics),
                         prepare_batch=prepare_batch, device=device)

        self.training = Engine(lambda e, b: self.train_batch(b))

        self._val_dataset = None

        self.training.epoch_started.add_handler(lambda e: self.lr_scheduler.step)

        @self.training.epoch_completed.add_handler
        def evaluate(e):
            if self._val_dataset is not None:
                self.evaluation.run(self.data_loader_f(self.validation_dataset))

    def attach_metric(self, metric, name=None):
        for eng in [self.training, self.evaluation]:
            super()._attach_metric(metric, name=name, engine=eng)

    def train(self, dataset, val_dataset=None):
        self._val_dataset = val_dataset
        return self.training.run(self.data_loader_f(dataset))

    def train_batch(self, batch):
        raise NotImplementedError()


# Supervised #######################################################################################

class SupervisedTrainer(Trainer):
    def eval_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = self.prepare_batch(batch, device=self.device)
            y_pred = self.model(x)
            return NameDict(prediction=y_pred, target=y)

    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch)
        from vidlu.nn import with_intermediate_outputs
        probs = self.model(x)
        # probs, f1 = with_intermediate_outputs(self.model, ['backbone.root'])(x)
        # print(f1)
        loss = self.loss(probs, y.long())
        loss.backward()
        # for n, p in self.model.named_parameters():
        #    print(p.requires_grad, p.grad.abs().mean(), n)
        self.optimizer.step()
        return NameDict(prediction=probs.argmax(1), probs=probs, target=y, loss=loss.item())


# Special

class ResNetCifarTrainer(SupervisedTrainer):
    # as in www.arxiv.org/abs/1603.05027
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        optimizer_f=partial(SGD, lr=1e-1, momentum=0.9),
        weight_decay=1e-4,
        epoch_count=200,
        lr_scheduler_f=partial(MultiStepLR, milestones=[60, 120, 160], gamma=0.2),
        batch_size=128)


class ResNetCamVidTrainer(SupervisedTrainer):
    # custom
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        optimizer_f=partial(SGD, lr=1e-1, momentum=0.9),
        weight_decay=1e-4,
        epoch_count=40,
        lr_scheduler_f=partial(MultiStepLR, milestones=[60, 120, 160], gamma=0.2),
        batch_size=32)


class WRNCifarTrainer(ResNetCifarTrainer):
    # as in www.arxiv.org/abs/1605.07146
    __init__ = partialmethod(ResNetCifarTrainer.__init__, weight_decay=5e-4)


class DenseNetCifarTrainer(SupervisedTrainer):
    # as in www.arxiv.org/abs/1608.06993
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        weight_decay=1e-4,
        optimizer_f=default_args(ResNetCifarTrainer).optimizer_f,
        epoch_count=100,
        lr_scheduler_f=partial(MultiStepLR, milestones=[50, 75], gamma=0.1),
        batch_size=64)


# Autoencoder ######################################################################################


class AutoencoderTrainer(Trainer):
    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x = self.prepare_batch(batch)[0]
        x_r = self.model(x)
        loss = self.loss(x_r, x)
        loss.backward()
        self.optimizer.step()
        return NameDict(reconstruction=x_r, input=x, loss=loss.item())


# GAN ##############################################################################################

class GANTrainer(Trainer):
    def __init__(self, model, loss=None, optimizer_f=None, epoch_count=1,
                 prepare_batch=engine._prepare_batch, device=None, non_blocking=False):
        loss = loss or nn.BCELoss()
        args = {k: v for k, v in locals() if k not in ['self', 'class']}
        super().__init__(**args)

    @lru_cache(1)
    def _get_real_labels(self, batch_size):
        return torch.ones(batch_size, device=self.model.device)

    @lru_cache(1)
    def _get_fake_labels(self, batch_size):
        return torch.zeros(batch_size, device=self.model.device)

    def train_batch(self, batch):
        self.model.train()
        """ Copied from ignite/examples/gan/dcgan and modified"""
        real = self.prepare_batch(batch)[0]
        batch_size = real.shape[0]
        real_labels = self._get_real_labels(batch_size)

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) ##########
        discriminator, generator = self.model.discriminator, self.model.generator
        discriminator.zero_grad()

        # training discriminator with real
        output = discriminator(real)
        errD_real = self.loss(output, real_labels)
        D_real = output.mean().item()
        errD_real.backward()

        fake = generator(self.model.sample_z(batch_size))

        # training discriminator with fake
        output = discriminator(fake.detach())
        errD_fake = self.loss(output, self._get_fake_labels(batch_size))
        D_fake1 = output.mean().item()
        errD_fake.backward()

        self.optimizer['D'].step()

        # (2) Update G network: maximize log(D(G(z))) ##########################
        generator.zero_grad()

        # Update generator. We want to make a step that will make it more likely that D outputs "real"
        output = discriminator(fake)
        errG = self.loss(output, real_labels)
        D_fake2 = output.mean().item()

        errG.backward()
        self.optimizer['G'].step()

        return NameDict(errD=(errD_real + errD_fake).item(), errG=errG.item(), D_real=D_real,
                        D_fake1=D_fake1, D_fake2=D_fake2)


# Default arguments ################################################################################

def get_default_argtree(trainer_class, model, dataset):
    from vidlu.problem import Problem, dataset_to_problem

    problem = dataset_to_problem(dataset)
    if issubclass(trainer_class, SupervisedTrainer):
        if problem == Problem.CLASSIFICATION:
            return ArgTree(loss_f=nn.CrossEntropyLoss)
        elif problem == Problem.SEMANTIC_SEGMENTATION:
            return ArgTree(loss_f=nn.CrossEntropyLoss)
    return ArgTree()
