from functools import partial, partialmethod, lru_cache

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from ignite import engine
from ignite.engine import Engine

from vidlu.data import DataLoader
from vidlu.utils.func import default_args
from vidlu.utils.collections import NameDict
from vidlu.utils.torch import prepare_batch
from vidlu.training.engine import Engine
from vidlu.nn.modules import get_device


# Runner ###########################################################################################

class Evaluator:
    def __init__(self, model, loss_f,
                 data_loader_f=partial(DataLoader, batch_size=1, num_workers=2),
                 prepare_batch=prepare_batch, device=None):
        super().__init__()
        self.model = model
        self.loss = loss_f()
        self.device = device
        self.prepare_batch = partial(prepare_batch, device=device, non_blocking=False)
        self.data_loader_f = data_loader_f

        self.evaluation = Engine(lambda e, b: self.eval_batch(b))
        self.metrics = {}

    def _attach_metric(self, metric, name=None, engine=None):
        name = name or type(metric).__name__
        self.metrics[name] = metric
        engine = engine or self.evaluation
        engine.epoch_started.add_handler(lambda e: metric.reset())
        engine.iteration_completed.add_handler(lambda e: metric.update(e.state.output))

        def compute_metric(e):
            e.state.metrics[name] = metric.compute()

        engine.iteration_completed.add_handler(lambda e: compute_metric(e))

    def attach_metric(self, metric, name=None):
        self._attach_metric(metric, name=name, engine=self.evaluation)

    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def get_metrics(self, *, reset=False):
        metric_evals = dict()
        for k, m in self.metrics.items():
            value = m.compute()
            metric_evals.update(value if isinstance(value, dict) else {k: value.compute()})
        if reset:
            self.reset_metrics()
        return metric_evals

    def get_outputs(self, *x):
        if hasattr(self.model, 'get_outputs'):
            return self.model.get_outputs(*x)
        prediction = self.model(*x)
        return prediction, NameDict(prediction=prediction)

    def eval(self, dataset):
        return self.evaluation.run(self.data_loader_f(dataset))

    def eval_batch(self, batch):
        raise NotImplementedError()


# Trainer ##########################################################################################

class Trainer(Evaluator):
    def __init__(self, model, loss_f, weight_decay, optimizer_f, epoch_count,
                 lr_scheduler_f=partial(LambdaLR, lr_lambda=lambda e: 1), batch_size=1,
                 data_loader_f=default_args(Evaluator).data_loader_f,
                 prepare_batch=prepare_batch, device=None):
        super().__init__(model, loss_f=loss_f,
                         data_loader_f=partial(data_loader_f, batch_size=batch_size),
                         prepare_batch=prepare_batch,
                         device=get_device(model) if device is None else device)
        self.optimizer = optimizer_f(params=self.model.parameters(), weight_decay=weight_decay)
        self.epoch_count = epoch_count
        self.lr_scheduler = lr_scheduler_f(optimizer=self.optimizer)

        self.training = Engine(lambda e, b: self.train_batch(b))

        self.training.epoch_started.add_handler(lambda e: self.lr_scheduler.step)

    def attach_metric(self, metric, name=None):
        for eng in [self.training, self.evaluation]:
            super()._attach_metric(metric, name=name, engine=eng)

    def train(self, dataset):
        data_loader = self.data_loader_f(dataset)
        return self.training.run(data_loader, max_epochs=self.epoch_count)

    def train_batch(self, batch):
        raise NotImplementedError()


# Supervised #######################################################################################

class SupervisedTrainer(Trainer):
    def eval_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = self.prepare_batch(batch, device=self.device)

            prediction, outputs = self.get_outputs(x)
            #print('.')
            #on_cuda = True
            #with torch.autograd.profiler.profile(use_cuda=on_cuda) as prof:
            #    prediction, outputs = self.get_outputs(x)
            #if on_cuda:
            #    torch.cuda.synchronize()
            #print(prof.key_averages().table('cuda_time_total'))
            #exit()

            prediction, outputs = self.get_outputs(x)
            loss = self.loss(prediction, y.long())
            return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item())

    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch, device=self.device)

        prediction, outputs = self.get_outputs(x)

        loss = self.loss(prediction, y.long())
        loss.backward()
        self.optimizer.step()
        return NameDict(prediction=prediction, target=y, outputs=outputs, loss=loss.item())


# Classification ###################################################################################

class ClassificationTrainer(SupervisedTrainer):
    def get_outputs(self, *x):
        if hasattr(self.model, 'get_outputs'):
            return self.model.get_outputs(*x)
        log_probs = self.model(*x)
        return log_probs, NameDict(prediction=log_probs, log_probs=log_probs, probs=log_probs.exp(),
                                   hard_prediction=log_probs.argmax(1))


# Special

class ResNetCifarTrainer(ClassificationTrainer):
    # as in www.arxiv.org/abs/1603.05027
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        optimizer_f=partial(SGD, lr=1e-1, momentum=0.9),
        weight_decay=1e-4,
        epoch_count=200,
        lr_scheduler_f=partial(MultiStepLR, milestones=[60, 120, 160], gamma=0.2),
        batch_size=128)


class ResNetCamVidTrainer(ClassificationTrainer):
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


class DenseNetCifarTrainer(ClassificationTrainer):
    # as in www.arxiv.org/abs/1608.06993
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        weight_decay=1e-4,
        optimizer_f=default_args(ResNetCifarTrainer).optimizer_f,
        epoch_count=100,
        lr_scheduler_f=partial(MultiStepLR, milestones=[50, 75], gamma=0.1),
        batch_size=64)


class SmallImageClassifierTrainer(ClassificationTrainer):
    # as in www.arxiv.org/abs/1603.05027
    __init__ = partialmethod(
        SupervisedTrainer.__init__,
        optimizer_f=partial(SGD, lr=1e-2, momentum=0.9),
        weight_decay=0,
        epoch_count=50,
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
        """ Copied from ignite/examples/gan/dcgan and modified"""
        self.model.train()
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
