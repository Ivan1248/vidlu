import typing as T
from dataclasses import dataclass

import torch

import vidlu.modules.inputwise  as vmi
import vidlu.modules as vm
from vidlu.utils.func import Required, partial


def _init_pert_model(pert_model, x, initializer=None):
    pert_model.eval()
    vm.call_if_not_built(pert_model, x)  # parameter shapes might have to be inferred from x
    if initializer is not None:
        with torch.no_grad():
            initializer(pert_model, x)
    pert_model.train()


@dataclass
class BasePerturber:
    def __call__(self, x, *args, **kwargs):
        """Returns a perturbation."""
        return self._get_perturbation(x, *args, **kwargs)

    def perturb(self, x, *args, **kwargs):
        """Perturbs the inputs.

        __call__ or perturb should be overridden by subclasses.

        Args:
            x (Tensor): Input tensor.
            *args: Other inputs.
            **kwargs: Keyword arguments to be forwarded.

        Return:
            Perturbed inputs.
        """
        perturb = self(x, *args, **kwargs)
        return perturb(x, *args)

    def _get_perturbation(self, x, *args, **kwargs):
        """Returns a random perturbation function."""
        # this or _perturb  has to be implemented in subclasses
        return NotImplemented


@dataclass
class AttackWrapperPerturber(BasePerturber):
    attack: 'Attack'

    def _get_perturbation(self, x, *args, **kwargs):
        return self.attack(lambda x: x, x)


def attack_as_perturber(attack):
    return AttackWrapperPerturber(attack)


@dataclass
class PertModelPerturber(BasePerturber):
    pert_model_f: vmi.PertModelBase = partial(vmi.Add, ())
    initializer: T.Callable[[vmi.PertModelBase, torch.Tensor], None] = Required

    def _get_perturbation(self, x, *args, pert_model=None, initialize_pert_model=True):
        if pert_model is None:
            pert_model = self.pert_model_f()
        _init_pert_model(pert_model, x,
                         initializer=self.initializer if initialize_pert_model else None)
        return pert_model
