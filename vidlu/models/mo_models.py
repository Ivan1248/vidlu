from torch import nn

from vidlu.libs.swiftnet.models import semseg
from vidlu.libs.swiftnet.models.resnet.resnet_single_scale import resnet18

from vidlu.models import SegmentationModel


class MoSwiftnetRN18(nn.Module):
    def __init__(self):
        super().__init__()
        from data.cityscapes import Cityscapes

        scale = 1
        mean = [73.15, 82.90, 72.3]
        std = [47.67, 48.49, 47.73]

        num_classes = Cityscapes.num_classes

        resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
        self.wrapped = semseg.SemsegModel(resnet, num_classes)

    def forward(self, *args, **kwargs):
        logits, _ = self.wrapped(*args, **kwargs)[0]
        return logits


