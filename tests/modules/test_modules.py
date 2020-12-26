import pytest

from vidlu.modules import *
from vidlu.modules.components import Baguette
from vidlu.utils.collections import NameDict
import weakref
import gc

torch.no_grad()


class TestModule:
    def test_not_built(self):
        m = Module()
        assert not m._built

    def test_store_arguments(self):
        class TModule(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.store_args()

        class TModule2(TModule):
            def __init__(self, a, *args, c='v', d=6, **kwargs):
                super().__init__()
                self.store_args()

        m1 = TModule(1, 2, 3, c='spam')
        assert m1.args == NameDict(args=(1, 2, 3), kwargs={'c': 'spam'})
        m2 = TModule2(1, 2, c='unladen', swallow=8)
        assert m2.args == NameDict(a=1, args=(2,), c='unladen', d=6, kwargs={'swallow': 8})

    def test_inverse(self):
        m = Func(lambda x: x * 2, lambda x: x // 2)
        assert m.inverse(m(2)) == 2
        assert m.inverse is m.inverse
        assert not m.is_inverse
        assert m.inverse.is_inverse
        assert m.inverse.inverse is m
        assert m.inverse.inverse.inverse is m.inverse

        modules = weakref.WeakSet({m, m.inverse})
        del m
        gc.collect()
        assert len(modules) == 0
        assert len(InvertibleMixin._inverses) == 0
        assert len(InvertibleMixin._module_to_inverse) == 0

    def test_inverse_error(self):
        class A(Module):
            def forward(self, x):
                return x * 0

        m = A()
        with pytest.raises(ModuleInverseError):
            m.inverse

    """
    def test_scoped_sequential(self):
        x = torch.ones(4, 522)
        inner = Seq(lin1=Linear(8), lin2=Seq(Linear(5)))
        M = Seq(inner=inner)
        M(x)
        assert M in [p for p, cname in inner.parents]
        assert inner in [p for p, cname in inner.lin1.parents]
    """


class TestFunc:
    def test_square(self):
        x = torch.ones(4, 1, 28, 28)
        node = Func(lambda x: x ** 2).eval()
        assert torch.all(node(x) == x ** 2)


class TestSequentialForkParallelReduce:
    def test_module_table(self):
        m = ModuleTable(a=Identity(), b=Identity())
        assert m['a'] is m[0]
        assert m['b'] is m[1]

    def test_seq(self):
        m = Seq(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        assert m['a'] is m[0]
        assert m['b'] is m[1]
        for i in range(100):
            assert m(torch.tensor(i)) == (i + 1) * 3

    def test_fork(self):
        m = Fork(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        for i in range(100):
            assert m(torch.tensor(i)) == ((i + 1), 3 * i)

    def test_parallel(self):
        m = Parallel(a=Func(lambda x: x + 1), b=Func(lambda x: x * 3))
        for i in range(100):
            assert (m((torch.tensor(i), torch.tensor(i + 1))) == (
                torch.tensor(i + 1), torch.tensor(3 * (i + 1))))

    def test_reduce(self):
        a = tuple(map(torch.tensor, range(5)))
        for m in [Reduce(lambda x, y: x.add_(y)), Sum()]:
            assert m(a) == sum(a)

    def test_combined(self):
        m = Seq(fork=Fork(id=Identity(),
                          sqr=Func(lambda x: x ** 2)),
                para=Parallel(mul2=Func(lambda x: 2 * x),
                              mul3=Func(lambda x: x * 3)),
                sum=Sum())
        for i in range(5):
            assert m(torch.tensor(i)) == 2 * i + 3 * i ** 2

    def test_intermediate(self):
        m = Seq(fork=Fork(id=Identity(),
                          sqr=Func(lambda x: x ** 2)),
                para=Parallel(add2=Func(lambda x: x + 2),
                              mul3=Func(lambda x: x * 3)),
                sum=Sum())
        inter = ["fork.id", "para.mul3", "para"]
        iomw = with_intermediate_outputs(m, inter)
        for i in range(5):
            id_ = torch.tensor(i)
            add2 = id_ + 2
            mul3 = 3 * id_ ** 2
            para = (add2, mul3)
            sum_ = add2 + mul3
            assert iomw(torch.tensor(i)) == (sum_, (id_, mul3, para))


class TestConv:
    def test_build(self):
        x = torch.ones(4, 1, 28, 28)
        conv = Conv(2, 3, bias=True).eval()
        assert conv.args.in_channels is None
        conv(x)
        assert conv.args.in_channels == x.shape[1]

    def test_padding(self):
        for ksize in [1, 3, 5]:
            conv = Conv(53, ksize, padding=0, bias=False).eval()
            assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 53, 28 - ksize + 1, 28 - ksize + 1)

            conv = Conv(53, ksize, padding='half', bias=False).eval()
            assert conv(torch.ones(4, 1, 28, 28)).shape == (4, 53, 28, 28)

            conv = Conv(53, ksize, padding='full', bias=False).eval()
            assert conv(torch.ones(4, 2, 28, 28)).shape == (4, 53, 28 + ksize - 1, 28 + ksize - 1)

    def test_conv_dim(self):
        conv = Conv(8, 3, bias=True).eval()
        conv(torch.randn(1, 2, 3))
        assert isinstance(conv.orig, nn.Conv1d)

        conv = Conv(8, 3, bias=True).eval()
        conv(torch.randn(1, 2, 3, 4))
        assert isinstance(conv.orig, nn.Conv2d)

        conv = Conv(8, 3, bias=True).eval()
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


class TestBaguette:
    def test_baguette(self):
        bk = Baguette(2)
        assert bk.inverse is bk.inverse
        x = torch.arange(2 * 4 * 8 * 6).view(2, 4, 8, 6)
        assert torch.all(bk.inverse(bk(x)) == x)


class TestSplit:
    def test_split(self):
        x = torch.zeros(2, 5)
        split = Split(2)
        a, b, c = split(x)
        assert torch.all(b.eq(x[:, 2:4]))

    def test_split_ellipsis(self):
        x = torch.zeros(1, 5)
        split1 = Split([2, ...])
        split2 = Split([2, 3])
        a1, b1 = split1(x)
        a2, b2 = split2(x)
        assert torch.all(a1.eq(a2))
        assert torch.all(b1.eq(b2))


class TestRestruct:
    def test_restruct(self):
        restruct = Restruct("(a, b),c,( d )", "((a) ,b), (c, d)")
        x = ((1, 2), 3, (4,))
        y = (((1,), 2), (3, 4))
        assert restruct(x) == y
        assert restruct.inverse(y) == x

    def test_restruct_error(self):
        x = ((1, 2), (3,))
        restruct = Restruct("((a, b), (c))", "((a), b, (c))")
        with pytest.raises(TypeError):
            restruct(x)
        restruct = Restruct("(a, b), (c)", "(a), b, (c)")
        assert restruct(x) == ((1,), 2, (3,))


class TestDeepSplit:
    def test_deep_split(self):
        module = Seq(a=Linear(8),
                     b=Seq(
                         a=Seq(
                             a=Linear(9),
                             b=Fork(Linear(10), Linear(5)),
                             c=Concat()),
                         b=Linear(12)),
                     c=Seq(
                         a=Linear(13)))
        x = torch.randn((4, 8))
        module(x)
        for path in ['', 'a', 'b', 'c', 'b.a', 'b.b', 'c.a', 'b.a.b', ]:
            left, right = deep_split(module, path)
            rejoined = deep_join(left, right)
            assert str(rejoined) == str(module)
            assert torch.all(rejoined(x) == module(x))
