"""Optional compatibility helpers for bridging Vidlu Trainer ↔ PyTorch Lightning."""
import typing as T

import torch

from vidlu.optim.lr_schedulers import ConstLR
from vidlu.utils.collections import NameDict
from vidlu.training.trainers import Trainer


def to_lightning_trainer(trainer):
    import pytorch_lightning as pl

    class PLModule(pl.LightningModule):
        def __init__(self, vidlu_trainer):
            super().__init__()
            # Expose the underlying model as a submodule so Lightning moves it across devices
            self.model = vidlu_trainer.model
            self._vidlu_trainer = vidlu_trainer
            # Use manual optimization since Vidlu steps its own optimizer inside train steps
            self.automatic_optimization = False

        def forward(self, x):
            return self.model(x)

        def _prepare(self, batch):
            # Reuse Vidlu's batch preparation (device placement, etc.)
            return self._vidlu_trainer.prepare_batch(batch)

        def _log_result(self, result_dict, prefix):
            if isinstance(result_dict, dict):
                for k, v in result_dict.items():
                    if isinstance(v, (int, float)):
                        self.log(f"{prefix}/{k}", v, prog_bar=(k == 'loss'), on_step=True,
                                 on_epoch=True, batch_size=None)

        def training_step(self, batch, batch_idx):
            batch = self._prepare(batch)
            result = self._vidlu_trainer.train_step(self._vidlu_trainer, batch)
            self._log_result(result, "train")
            # Return a tensor loss for Lightning's hooks even though optimization is manual
            loss_value = result.get('loss', 0.0)
            return torch.as_tensor(loss_value, device=self.device, dtype=torch.float32)

        def validation_step(self, batch, batch_idx):
            batch = self._prepare(batch)
            result = self._vidlu_trainer.eval_step(self._vidlu_trainer, batch)
            self._log_result(result, "val")
            loss_value = result.get('loss', 0.0)
            return torch.as_tensor(loss_value, device=self.device, dtype=torch.float32)

        def configure_optimizers(self):
            # Note: with manual_optimization=True, Lightning ignores this for stepping,
            # but may still use it for scheduling hooks. Returning it for completeness.
            return [self._vidlu_trainer.optimizer], [self._vidlu_trainer.lr_scheduler]

    return PLModule(trainer)


def from_lightning_trainer(lightning_module, *, epoch_count=1, batch_size=1, eval_batch_size=None,
                           deterministic=False, distributed=False):
    """Adapt a LightningModule to a Vidlu Trainer.

    If the Lightning module was created via to_lightning_trainer, unwraps and returns the
    original Vidlu Trainer. Otherwise, creates a minimal adapter Trainer that calls the
    Lightning module's training_step/validation_step under Vidlu's loops.
    """
    # If it's our wrapper, return the original Vidlu trainer
    if hasattr(lightning_module, "_vidlu_trainer"):
        return lightning_module._vidlu_trainer

    # Extract optimizer/scheduler if available
    def _extract_opt_sch(lm):
        try:
            cfg = lm.configure_optimizers()
        except Exception:
            return None, None
        # Possible returns per Lightning API:
        # - optimizer
        # - (optimizer, scheduler)
        # - ([optimizers], [schedulers])
        # - dicts; we only support basic cases here
        opt, sch = None, None
        if isinstance(cfg, tuple):
            if len(cfg) == 2 and isinstance(cfg[0], list):
                opt = cfg[0][0] if cfg[0] else None
                sch = cfg[1][0] if cfg[1] else None
            elif len(cfg) == 2:
                opt, sch = cfg
        elif isinstance(cfg, list):
            opt = cfg[0] if cfg else None
        else:
            opt = cfg
        return opt, sch

    opt, sch = _extract_opt_sch(lightning_module)

    # Create minimal wrappers around Lightning's steps using closures (lm captured from scope)
    lm = lightning_module

    def _train_step(vtrainer, batch):
        lm.train()
        out = lm.training_step(batch, getattr(vtrainer.training.state, 'abs_iteration', 0))
        loss = out['loss'] if isinstance(out, dict) and 'loss' in out else out
        # Perform optimization here if an optimizer is present and loss is a tensor
        if opt is not None and isinstance(loss, torch.Tensor):
            vtrainer.optimizer.zero_grad()
            loss.backward()
            vtrainer.optimizer.step()
        loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        return NameDict(loss=loss_value)

    @torch.no_grad()
    def _eval_step(vtrainer, batch):
        lm.eval()
        validation_step = getattr(lm, 'validation_step', None)
        if validation_step is None:
            # Fallback to forward-only if no validation_step
            y = lm.forward(batch[0]) if isinstance(batch, (tuple, list)) else lm.forward(batch)
            return NameDict(out=y, loss=-1.0)
        out = validation_step(batch, getattr(vtrainer.evaluation.state, 'abs_iteration', 0))
        loss = out['loss'] if isinstance(out, dict) and 'loss' in out else out
        loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        return NameDict(loss=loss_value)

    # Optimizer and scheduler factories that return prebuilt instances
    def optimizer_f(params):
        return opt if opt is not None else torch.optim.SGD(params, lr=0.0)

    def lr_scheduler_f(**kwargs):
        return sch if sch is not None else ConstLR(optimizer=kwargs['optimizer'])

    def _dummy_loss(out, y, reduction="mean"):
        if isinstance(out, torch.Tensor):
            return torch.zeros((), device=out.device)
        return torch.zeros(())

    return Trainer(
        model=lm,
        loss=_dummy_loss,
        eval_step=_eval_step,
        train_step=_train_step,
        epoch_count=epoch_count,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size if eval_batch_size is not None else batch_size,
        optimizer_f=optimizer_f,
        lr_scheduler_f=lr_scheduler_f,
        deterministic=deterministic,
        distributed=distributed,
    )
