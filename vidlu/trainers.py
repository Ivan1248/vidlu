from functools import wraps, partial, partialmethod, lru_cache

import torch
from torch import nn
from ignite import engine
from torch import optim
from ignite.engine import Engine, Events
from ignite._utils import convert_tensor

from vidlu.data import DataLoader

from vidlu.utils.misc import Event
from vidlu.utils.func import Empty, Full
from vidlu.utils.collections import NameDict
from vidlu.nn.utils import with_intermediate_outputs



def _prepare_batch(batch, device=None, non_blocking=False):
    return tuple(convert_tensor(x, device=device, non_blocking=non_blocking) for x in batch)


# Trainer ##########################################################################################

def get_trainer(trainer_f, model, training_config, **kwargs):
    tc = training_config
    return trainer_f(model, loss=tc.loss, optimizer=tc.lr_scheduler, epoch_count=tc.lr_scheduler,
                     **kwargs)


class Trainer(Engine):
    def __init__(self, model, loss_f, optimizer_f, weight_decay, lr_scheduler_f, epoch_count,
                 data_loader_f=partial(DataLoader, batch_size=1), prepare_batch=_prepare_batch,
                 device=None):
        super().__init__(type(self).train_batch)
        self.model = model
        self.loss, self.optimizer, self.epoch_count = loss_f(), optimizer_f, epoch_count
        self.prepare_batch = partial(prepare_batch, device=None, non_blocking=False)

        # events
        self.started = self.on(Events.STARTED, Event())
        self.completed = self.on(Events.COMPLETED, Event())
        self.epoch_started = self.on(Events.EPOCH_STARTED, Event())
        self.epoch_completed = self.on(Events.EPOCH_COMPLETED, Event())
        self.iteration_started = self.on(Events.ITERATION_STARTED, Event())
        self.iteration_completed = self.on(Events.ITERATION_COMPLETED, Event())

    def train_batch(self, batch):
        raise NotImplementedError()


# Supervised #######################################################################################

class SupervisedTrainer(Trainer):
    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch)
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return NameDict(prediction=y_pred, target=y, loss=loss.item())


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
        real = self.prepare_batch(batch)[0]
        batch_size = real.shape[0]
        real_labels = self._get_real_labels(batch_size)

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) ##########
        discriminator, generator = self.model.discriminator, self.model.generator
        discriminator.zero_grad()

        # train discriminator with real
        output = discriminator(real)
        errD_real = self.loss(output, real_labels)
        D_real = output.mean().item()
        errD_real.backward()

        fake = generator(self.model.sample_z(batch_size))

        # train discriminator with fake
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
