import torch
import torchvision

from vidlu import factories, problem, parameters
from vidlu.utils import text

torch.no_grad()


def compare_with_torchvision(mymodelname, tvmodelname):
    x = torch.randn(1, 3, 64, 64)
    vidlu_model = factories.get_model(mymodelname,
                                      problem=problem.Classification(class_count=8),
                                      init_input=x,
                                      verbosity=2)
    tv_model = torchvision.models.__dict__[tvmodelname](num_classes=8)

    tsd = tv_model.state_dict()

    params = parameters.translate(
        text.scan("{a:([a-zA-Z]+)}(\d+)", tvmodelname)['a'], tsd)
    vidlu_model.load_state_dict(params)

    assert torch.all(vidlu_model(x) == tv_model(x))


class TestResNet:
    def test_resnet(self):
        compare_with_torchvision("ResNetV1,backbone_f=t(depth=18,small_input=False)", "resnet18")


class TestDenseNet:
    def test_densenet(self):
        compare_with_torchvision("DenseNet,backbone_f=t(depth=121,small_input=False)",
                                 "densenet121")
