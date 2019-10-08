from functools import partial

from torch import nn

from vidlu.modules.elements import (Conv, ConvTranspose, BatchNorm)

__all__ = ['norm_f', 'act_f', 'conv_f', 'convt_f']

norm_f = BatchNorm
act_f = nn.ReLU  # inplace=True can cause bugs
conv_f = partial(Conv, padding='half', bias=False)
convt_f = partial(ConvTranspose, bias=False)
