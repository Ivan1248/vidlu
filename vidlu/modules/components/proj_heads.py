import vidlu.modules.elements as E
from vidlu.utils.func import partial


class ProjHead(E.Seq):
    lin_name = 'lin'

    def __init__(self, dims=(None,) * 2, lin_f=partial(E.Linear, bias=False), norm_f=E.BatchNorm,
                 act_f=E.ReLU):
        super().__init__()
        if any(d is None for d in dims):
            self.args = self.get_args(locals())
        else:
            self._build(dims, lin_f, norm_f, act_f)
            self._built = True

    def build(self, x):
        # If a dimension is None, set it to the dimension of the input
        self.args.dims = [x.shape[1] if d is None else d for d in self.args.dims]
        self._build(**self.args)
        del self.args

    def _build(self, dims, lin_f, norm_f, act_f):
        for i, dim in enumerate(dims[:-1]):
            self.add(f"{self.lin_name}{i}", lin_f(dim))
            if norm_f is not None:
                self.add(f"norm{i}", norm_f())
            self.add(f"act{i}", act_f())
        self.add(f"{self.lin_name}{len(dims) - 1}", lin_f(dims[-1]))


class ConvProjHead(ProjHead):
    lin_name = 'conv'

    def __init__(self, dims=(None,) * 2,
                 conv_f=partial(E.Conv, kernel_size=1, stride=1, bias=False), norm_f=E.BatchNorm,
                 act_f=E.ReLU):
        super().__init__(dims, lin_f=conv_f, norm_f=norm_f, act_f=act_f)


class BarlowTwinsProjHead(E.Seq):  # From BarlowTwins and VICReg
    def __init__(self, dims=(8192,) * 3, lin_f=partial(E.Linear, bias=False), norm_f=E.BatchNorm,
                 act_f=E.ReLU):
        super().__init__()
        for i, dim in enumerate(dims[:-1]):
            self.add(**{f"linear{i}": lin_f(),
                        f"norm{i}": norm_f(),
                        f"act{i}": act_f()})
        self.add(lin_f(dims[-1], bias=False))
