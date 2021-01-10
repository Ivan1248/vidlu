from vidlu.utils.func import partial

from torch import nn

from vidlu.utils.func import Reserved

from .models import Model
from . import initialization


class MoSwiftnetRN18(Model):
    def __init__(self, input_adapter=None):
        from vidlu.libs.swiftnet.data.cityscapes import Cityscapes
        from vidlu.libs.swiftnet.models import semseg
        from vidlu.libs.swiftnet.models.resnet.resnet_single_scale import resnet18

        super().__init__(init=lambda *a, **k: None)

        scale = 1
        mean = [73.15, 82.90, 72.3]
        std = [47.67, 48.49, 47.73]

        #mean = [0.] * 3
        #std = [1.] * 3

        num_classes = Cityscapes.num_classes

        resnet = resnet18(pretrained=True, efficient=True, mean=mean, std=std, scale=scale)
        self.input_adapter = input_adapter
        self.wrapped = semseg.SemsegModel(resnet, num_classes)

    def forward(self, x):
        # self.wrapped.backbone.img_mean.fill_(1.)
        self.wrapped.backbone.img_mean.fill_(0.)
        self.wrapped.backbone.img_std.fill_(1.)
        #self.wrapped.backbone.img_std.view(-1)[0] = 1.  # fill_(1.)
        from vidlu.libs.swiftnet.models.resnet.resnet_single_scale import check
        check(x, "input")
        x = self.input_adapter(x)
        image_size = x.shape[-2:]
        logits = self.wrapped(x, target_size=image_size, image_size=image_size)[0]
        return logits
