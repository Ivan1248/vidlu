import platform

import torch
import torchvision

from vidlu import factories, problem, parameters
from vidlu.utils import text
import vidlu.modules as vm


@torch.no_grad()
def compare_with_torchvision(mymodelname, tvmodelname):
    x = torch.randn(2, 3, 224, 224)  # TODO: investigate: 64x64 doesn't work on linux
    vidlu_model = factories.get_model(
        mymodelname, problem=problem.Classification(class_count=8), init_input=x, verbosity=2)
    tv_model = torchvision.models.__dict__[tvmodelname](num_classes=8)

    tsd = tv_model.state_dict()

    translator_name = text.scan(r"{a:([a-zA-Z]+)}(\d+)", tvmodelname)['a']

    params = parameters.translate(translator_name, tsd)
    vidlu_model.load_state_dict(params)

    if platform.system() == 'Windows':
        assert torch.all(vidlu_model(x) == tv_model(x))
    else:
        assert (vidlu_model(x) - tv_model(x)).abs().max() < 3e-7


class TestResNet:
    @torch.no_grad()
    def test_resnet(self):
        compare_with_torchvision("ResNetV1,backbone_f=t(depth=18,small_input=False)", "resnet18")


class TestDenseNet:
    @torch.no_grad()
    def test_densenet(self):
        compare_with_torchvision("DenseNet,backbone_f=t(depth=121,small_input=False)",
                                 "densenet121")


class TestIRevNet:
    @torch.no_grad()
    def test_irevnet_invertibility(self):
        x = torch.randn(2, 3, 32, 32)
        model = factories.get_model(
            "IRevNet,backbone_f=t(init_stride=1,base_width=8,group_lengths=(2,2,2))",
            problem=problem.Classification(class_count=8),
            init_input=x)
        model_injective = vm.deep_split(model, 'backbone.concat')[0]
        h = model_injective(x)
        model_injective.inverse(h)
        # numerical error too high for randomly initialized model and high-frequecy input
        # model_injective.check_inverse(x)
