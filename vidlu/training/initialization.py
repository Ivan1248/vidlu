import warnings

from torch import nn

from vidlu.modules import components as com
from vidlu.modules import elements as mod


def kaiming_resnet(module, nonlinearity='relu', zero_init_residual=True):
    # from torchvision/models/resnet.py
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
    if zero_init_residual:
        found = False
        for m in module.modules():
            if isinstance(m, com.ResNetV2Unit):
                for i, c in enumerate(reversed(list(m.children()))):
                    if i == 1 and isinstance(c, mod.BatchNorm):
                        found = True
                        nn.init.constant_(c.orig.weight, 0)
        if not found:
            warnings.warn("Batch normalization modules for residual zero-init not found.")


def kaiming_densenet(module, nonlinearity='relu'):
    # from torchvision/models/densenet.py
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)  # added nonlinearity
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
