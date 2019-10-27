import torch.nn as nn

from vidlu.modules.elements import Seq, Conv, MaxPool, Linear, BatchNorm


class MNISTNetBackbone(Seq):
    def __init__(self, act_f=nn.ReLU, use_bn=False):
        super().__init__()
        add = self.add_module

        def add_conv_block(i, width):
            for j in range(2):
                add(f'conv{i}_{j}', Conv(width, 3, bias=not use_bn))
                if use_bn:
                    add(f'norm{i}_{j}', BatchNorm())
                add(f'act{i}_{j}', act_f())
                add(f'pool{i}', MaxPool(2, 2))

        def add_linear_block(i):
            for j in range(2):
                add(f'linear{i}', Linear(200, bias=not use_bn))
                if use_bn:
                    add(f'norm{i}_{j}', BatchNorm())
                add(f'act{i}_{j}', act_f())
                add(f'noise{i}_{j}', nn.Dropout(0.5))

        add_conv_block(0, 32)
        add_conv_block(1, 64)
        add_linear_block(2)
        add_linear_block(3)
