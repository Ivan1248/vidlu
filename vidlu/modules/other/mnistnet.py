import torch.nn as nn

from vidlu.modules.elements import Seq, Conv, MaxPool, Linear, BatchNorm


class MNISTNetBackbone(Seq):
    def __init__(self, act_f=nn.ReLU, use_bn=False):
        super().__init__()

        def add_conv_block(i, width):
            for j in range(2):
                self.add(f'conv{i}_{j}', Conv(width, 3, bias=not use_bn))
                if use_bn:
                    self.add(f'norm{i}_{j}', BatchNorm())
                self.add(f'act{i}_{j}', act_f())
                self.add(f'pool{i}', MaxPool(2, 2))

        def add_linear_block(i):
            for j in range(2):
                self.add(f'linear{i}', Linear(200, bias=not use_bn))
                if use_bn:
                    self.add(f'norm{i}_{j}', BatchNorm())
                self.add(f'act{i}_{j}', act_f())
                self.add(f'noise{i}_{j}', nn.Dropout(0.5))

        add_conv_block(0, 32)
        add_conv_block(1, 64)
        add_linear_block(2)
        add_linear_block(3)
