import warnings

from torch import nn

import vidlu.modules.components as vmc
import vidlu.modules as vm


def kaiming_resnet(module, nonlinearity='relu', zero_init_residual=False):
    # based on initialization from torchvision/models/resnet.py
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, vm.FastDeconv)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.reset_parameters()
            nn.init.constant_(m.bias, 0)
        elif len(m._parameters) > 0:
            warnings.warn(f"kaiming_resnet initialization not defined for {name}: {type(m)}.")
    if zero_init_residual:
        found = 0
        for m in module.modules():
            if isinstance(m, vmc.ResNetV1Unit):  # does not work better for mc.ResNetV2Unit
                found += 1
                last_bn = [c for c in m.fork.block.children() if isinstance(c, vm.BatchNorm)][-1]
                nn.init.constant_(last_bn.orig.weight, 1e-16)
        if not found:
            warnings.warn("Batch normalization module for residual zero-init not found.")


def kaiming_swiftnet(module, nonlinearity='relu'):
    kaiming_resnet(module, nonlinearity=nonlinearity, zero_init_residual=False)


def kaiming_densenet(module, nonlinearity='relu'):
    # based on initialization from torchvision/models/densenet.py
    for name, m in module.modules():
        if 'Conv' in type(m).__name__ and hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.reset_parameters()
            nn.init.constant_(m.bias, 0)
        elif len(m._parameters) > 0:
            warnings.warn(f"kaiming_densenet initialization not defined for {name}: {type(m)}.")


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
