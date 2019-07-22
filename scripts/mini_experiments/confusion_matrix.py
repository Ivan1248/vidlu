from argparse import Namespace

import numpy as np

from _context import vidlu
from vidlu.libs.swiftnet.lib.cylib import collect_confusion_matrix

from sklearn.metrics import confusion_matrix

class_count = 3

cmcy = np.zeros((class_count, class_count), dtype=np.uint64)

true = np.array([0, 1, 2, 3, -1], dtype=np.uint32)
pred = np.array([0, 1, 1, 2, 0], dtype=np.uint32)

collect_confusion_matrix(pred, true, cmcy)
cmskl = confusion_matrix(true, pred, labels=np.arange(3))

print(cmcy == cmskl)
print(cmcy == cmskl.T)

from vidlu.training.metrics import ClassificationMetrics
import torch

clam = ClassificationMetrics(3)
clam.update(Namespace(target=torch.tensor(np.array(true, dtype=np.int32)),
                      outputs=Namespace(
                          hard_prediction=torch.tensor(np.array(pred, dtype=np.int32)))))
print(clam.compute(('mP', 'mR', 'mIoU')))
print((1 + 1 / 2 + 0) / 3, (1 + 1 + 0) / 3)
clam.cm = clam.cm.T
print(clam.compute(('mP', 'mR', 'mIoU')))
