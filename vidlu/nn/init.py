from torch import nn
from vidlu.nn import components


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
        for m in module.modules():
            if isinstance(m, components.ResGroups):
                nn.init.constant_(m.post_norm.orig.weight, 0)


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
