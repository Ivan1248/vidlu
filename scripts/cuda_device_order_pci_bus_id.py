import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for modulename in ['torch', 'tensorflow', 'mxnet', 'keras', 'chainer']:
    if modulename in sys.modules:
        raise ImportError(
            "CUDA_DEVICE_ORDER must be changed before a deep learning framework is loaded:" +
            f" {modulename} should be imported after {__name__}.")
