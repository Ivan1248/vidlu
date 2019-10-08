import os
import sys

for modulename in ['torch', 'tensorflow', 'mxnet', 'chainer']:
    if modulename in sys.modules:
        raise RuntimeError(f"{modulename} should be imported after {__name__}.")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # as opposed to "FASTEST_FIRST"
