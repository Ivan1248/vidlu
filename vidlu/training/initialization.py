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
        found = 0
        for m in module.modules():
            # break #
            if isinstance(m, (com.ResNetV1Unit)):
                # not com.ResNetV2Unit because it seems not to work better
                block = m.branching.block
                last_bn = [c for c in block.children() if isinstance(c, mod.BatchNorm)][-1]
                nn.init.constant_(last_bn.orig.weight, 1e-16)
        if not found:
            warnings.warn("Batch normalization modules for residual zero-init not found.")


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
