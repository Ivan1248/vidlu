import warnings

from torch import nn

import vidlu.modules.components as mc
import vidlu.modules as M


def kaiming_resnet(module, nonlinearity='relu', zero_init_residual=False):
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
        found = 0
        for m in module.modules():
            if isinstance(m, mc.ResNetV1Unit):
                found += 1
                # not mc.ResNetV2Unit because it seems not to work better
                last_bn = [c for c in m.fork.block.children() if isinstance(c, M.BatchNorm)][-1]
                nn.init.constant_(last_bn.orig.weight, 1e-16)
        if not found:
            warnings.warn("Batch normalization module for residual zero-init not found.")


def kaiming_densenet(module, nonlinearity='relu'):
    # from torchvision/models/densenet.py
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity=nonlinearity)  # added nonlinearity
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


def kaiming_mnistnet(module, nonlinearity='relu'):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
