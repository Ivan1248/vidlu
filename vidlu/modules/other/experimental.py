from vidlu.utils.func import partial

from torch import nn

from vidlu.modules.elements import Module
from vidlu.modules.components import PreactBlock


class SlugUnit(Module):
    def __init__(self, change_dim=False,
                 block_f=partial(PreactBlock, kernel_sizes=[1, 3], width_factors=[4, 1])):
        super().__init__()
        self.store_args()
        self.blocks = None

    def build(self, inputs):
        last_dim = inputs[-1].shape[2]
        stride_factor = 1 + int(self.args.change_dim)
        strides = [stride_factor * (1 + int(x.shape[2] > last_dim)) for x in inputs]
        self.blocks = nn.ModuleList([self.args.block_f(stride=s) for s in strides])

    def forward(self, inputs):
        return sum(b(x) for b, x in zip(self.blocks, inputs))
