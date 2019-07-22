import pytest

import torch
from torch import nn
import numpy as np

from vidlu.modules import (Module, Func, Conv, Linear, BatchNorm, Sequential, Branching, Parallel,
                           Reduce, Sum, IntermediateOutputsModuleWrapper)
from vidlu.utils.collections import NameDict

torch.no_grad()


class TestModule:
    def test_not_built(self):
        m = Module()
        assert not m._built

    def test_store_arguments(self):
        class TModule(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class TModule2(TModule):
            def __init__(self, a, *args, c='v', d=6, **kwargs):
                super().__init__()

        m1 = TModule(1, 2, 3, c='spam')
        assert m1.args == NameDict(args=(1, 2, 3), kwargs={'c': 'spam'})
        m2 = TModule2(1, 2, c='unladen', swallow=8)
        assert m2.args == NameDict(a=1, args=(2,), c='unladen', d=6, kwargs={'swallow': 8})

    """
    def test_scoped_sequential(self):
        x = torch.ones(4, 522)
        inner = Sequential(lin1=Linear(8), lin2=Sequential(Linear(5)))
        m = Sequential(inner=inner)
        m(x)
        assert m in [p for p, cname in inner.parents]
        assert inner in [p for p, cname in inner.lin1.parents]
    """


class TestFunc:
    def test_square(self):
        x = torch.ones(4, 1, 28, 28)
        node = Func(lambda x: x ** 2).eval()
        assert torch.all(node(x) == x ** 2)


class TestSequentialBranchingParallelReduce:
    def test_sequential(self):
        m = Sequential(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        for i in range(100):
            assert m(torch.tensor(i)) == (i + 1) * 3

    def test_branching(self):
        m = Branching(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        for i in range(100):
            assert m(torch.tensor(i)) == ((i + 1), 3 * i)

    def test_parallel(self):
        m = Parallel(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        for i in range(100):
            assert (m((torch.tensor(i), torch.tensor(i + 1))) == (
                torch.tensor(i + 1), torch.tensor(3 * (i + 1))))

    def test_reduce(self):
        l = list(map(torch.tensor, range(5)))
        for m in [Reduce(lambda x, y: x.add_(y)), Sum()]:
            assert m(l) == sum(l)

    def test_combined(self):
        m = Sequential(branch=Branching(id=nn.Identity(),
                                        sqr=Func(lambda x: x ** 2)),
                       para=Parallel(mul2=Func(lambda x: 2 * x),
                                     mul3=Func(lambda x: x * 3)),
                       sum=Sum())
        for i in range(5):
            assert m(torch.tensor(i)) == 2 * i + 3 * i ** 2

    def test_intermediate(self):
        m = Sequential(branch=Branching(id=nn.Identity(),
                                        sqr=Func(lambda x: x ** 2)),
                       para=Parallel(add2=Func(lambda x: x + 2),
                                     mul3=Func(lambda x: x * 3)),
                       sum=Sum())
        iamw = IntermediateOutputsModuleWrapper(m, ["branch.id", "para.mul3", "para"])
        for i in range(5):
            id = i
            add2 = i + 2
            mul3 = 3 * i ** 2
            para = (add2, mul3)
            sum_ = add2 + mul3
            assert iamw(torch.tensor(i)) == (sum_, (id, mul3, para))


class TestConv:
    def test_build(self):
        x = torch.ones(4, 1, 28, 28)
        conv = Conv(2, 3).eval()
        assert conv.args.in_channels is None
        conv(x)
        assert conv.args.in_channels == x.shape[1]

    def test_padding(self):
        for ksize in [1, 3, 5]:
            conv = Conv(53, ksize, padding=0).eval()
            assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 53, 28 - ksize + 1, 28 - ksize + 1)

            conv = Conv(53, ksize, padding='half').eval()
            assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 53, 28, 28)

            conv = Conv(53, ksize, padding='full').eval()
            assert conv(torch.ones(4, 2, 28, 28)).shape == (4, 53, 28 + ksize - 1, 28 + ksize - 1)

    def test_conv_dim(self):
        conv = Conv(8, 3).eval()
        conv(torch.randn(1, 2, 3))
        assert isinstance(conv.orig, nn.Conv1d)

        conv = Conv(8, 3).eval()
        conv(torch.randn(1, 2, 3, 4))
        assert isinstance(conv.orig, nn.Conv2d)

        conv = Conv(8, 3).eval()
        conv(torch.randn(1, 2, 3, 4, 5))
        assert isinstance(conv.orig, nn.Conv3d)


class TestLinear:
    def test_flatten(self):
        out_dims = 53
        in_dims = (2, 3, 5, 7, 11)
        linear = Linear(out_dims)
        y = linear(torch.ones(*in_dims))
        assert linear.orig.weight.shape == (out_dims, np.prod(in_dims[1:]))
        assert y.shape == (in_dims[0], out_dims)


class TestBatchNorm:
    def test_bn(self):
        bn = BatchNorm().eval()
        bn(torch.randn(4, 2, 3))
        assert isinstance(bn.orig, nn.BatchNorm1d)

        bn = BatchNorm().eval()
        bn(torch.randn(4, 2, 3, 4))
        assert isinstance(bn.orig, nn.BatchNorm2d)

        bn = BatchNorm().eval()
        bn(torch.randn(4, 2, 3, 4, 5))
        assert isinstance(bn.orig, nn.BatchNorm3d)
