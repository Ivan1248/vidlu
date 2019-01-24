import os
import argparse
import re

import numpy as np

from _context import vidlu

from dl_uncertainty import dirs
from dl_uncertainty.utils.parsing import parse_log
from dl_uncertainty.utils.visualization import plot_curves
"""
/home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-15-0805/Model.log
/home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-18-0725/Model.log
/home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-18-0010/Model.log
"""

parser = argparse.ArgumentParser()
parser.add_argument('logpath', type=str)
args = parser.parse_args()

log_lines = open(args.logpath, 'r').readlines()
curves = parse_log(log_lines)
plot_curves(curves)
