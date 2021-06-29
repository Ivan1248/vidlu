import pytest
import contextlib as ctx

import torch
import torchvision
from tqdm import tqdm

from vidlu import factories
from vidlu import models
from vidlu.factories import problem
from vidlu.utils import text
import vidlu.modules as vm
import vidlu.modules.utils as vmu
from vidlu.modules.tensor_extra import LogAbsDetJac


@torch.no_grad()
def compare_with_torchvision(mymodelname, tvmodelname):
    torch.random.manual_seed(53)
    x = torch.randn(2, 3, 224, 224)  # TODO: investigate: 64x64 doesn't work on linux
    model_vidlu = factories.get_model(
        mymodelname, problem=problem.Classification(class_count=8), init_input=x, verbosity=2)
    assert vm.is_built(model_vidlu, thorough=True)
    if tvmodelname.startswith("resnet_v2"):
        import vidlu.modules.other as vmo
        model_tv = vmo.resnet_v2.__dict__[tvmodelname](num_classes=8)
    else:
        model_tv = torchvision.models.__dict__[tvmodelname](num_classes=8)

    model_vidlu.eval()
    model_tv.eval()

    tsd = model_tv.state_dict()
    translator_name = text.scan(r"{a:([\w_]*[a-zA-Z_])}(\d+)", tvmodelname, full_match=True)['a']
    translator_name = translator_name.rstrip("_")
    params = models.params.translate(translator_name, tsd)

    model_vidlu.load_state_dict(params)

    y_vidlu, interm_vidlu = vm.with_intermediate_outputs(model_vidlu, return_dict=True)(x)
    y_tv, interm_tv = vm.with_intermediate_outputs(model_tv, return_dict=True)(x)

    interm_translations = dict()
    interm_tv_transl = dict()
    from vidlu.utils.text import NoMatchError
    for k, v in interm_tv.items():
        try:
            d = models.params.translate(translator_name, {k: v})
            interm_tv_transl.update(d)
            interm_translations[k] = next(iter(d.keys()))
        except NoMatchError as e:
            pass

    def compare(a, b, k):
        if a.device == torch.device('cpu'):
            assert torch.all(a == b), k
        else:
            assert (a - b).abs().max() < 3e-7, k

    for k, v in tqdm(interm_tv_transl.items()):
        try:
            if v is None:
                breakpoint()
            compare(v, interm_vidlu[k], k)
        except AssertionError as e:
            raise e
    compare(y_vidlu, y_tv, k)


class TestResNet:
    @torch.no_grad()
    def test_resnet(self):
        # basic block
        compare_with_torchvision(
            "ResNetV1, backbone_f=t(depth=18, small_input=False, block_f=t(act_f=t(inplace=True)), groups_f=t(unit_f=t(inplace_add=True)))",
            "resnet18")

    @torch.no_grad()
    def test_resnet_bottleneck(self):
        compare_with_torchvision(
            "ResNetV1, backbone_f=t(depth=50, small_input=False, block_f=t(act_f=t(inplace=True)), groups_f=t(unit_f=t(inplace_add=True)))",
            "resnet50")


class TestResNetV2:
    @torch.no_grad()
    def test_resnet_v2(self):
        # basic block
        compare_with_torchvision(
            "ResNetV2, backbone_f=t(depth=18, small_input=False, block_f=t(act_f=t(inplace=True)), groups_f=t(unit_f=t(inplace_add=False)))",
            "resnet_v2_18")


class TestDenseNet:
    @torch.no_grad()
    def test_densenet(self):
        compare_with_torchvision("DenseNet,backbone_f=t(depth=121,small_input=False)",
                                 "densenet121")


class TestIRevNet:
    @torch.no_grad()
    def test_irevnet_invertibility_and_ladj(self):
        x = torch.randn(2, 3, 32, 32)
        LogAbsDetJac.set(x, LogAbsDetJac.zero(x))
        model = factories.get_model(
            "IRevNet,backbone_f=t(init_stride=4,base_width=None,group_lengths=(2,1))",
            problem=problem.Classification(class_count=8),
            init_input=x)
        assert vm.is_built(model, thorough=True)
        model_injective = vm.deep_split(model, 'backbone.concat')[0]
        assert vm.is_built(model_injective, thorough=True)

        h = model_injective(x)
        inverse_built = vm.is_built(model_injective.inverse)
        assert not inverse_built
        assert vm.is_built(model_injective.inverse, thorough=True) == inverse_built

        for keep_inv_ref in [True, False]:
            inv_ref = model_injective.inverse if keep_inv_ref else None
            model_injective.inverse(h)
            assert (model_injective.inverse is inv_ref) == keep_inv_ref
            assert vm.is_built(model_injective.inverse) == keep_inv_ref

        # numerical error too high for randomly initialized model and high-frequecy input
        # model_injective.check_inverse(x)
        for should_error in [False, True]:
            for m in (model if should_error else model_injective).modules():
                m.register_forward_hook(vmu.hooks.check_propagates_log_abs_det_jac)
            with pytest.raises(RuntimeError) if should_error else ctx.suppress():
                out, z = vm.with_intermediate_outputs(model, 'backbone.concat')(x)
            ladj = LogAbsDetJac.get(z)()
            assert torch.all(ladj == 0) and ladj.shape == (len(x),)
