import numpy as np

import matplotlib.pyplot as plt

# accur = np.array(
#  [.869, .896, .872, .891, .877, .896, .871, .890, .875, .895, .882, .899, .888, .932, 0.917, 0.918, .953])
# robus = np.array(
#  [.488, .469, .485, .471, .473, .463, .480, .475, .473, .460, .467, .450, .418, .001, 0.382, 0.357, .000])
# label = np.array(
#  ["A", "NTA", "A", "NTA", "A", "NTA", "A", "NTA", "A", "NTA", "A", "NTA", "A", "NTA", "A", "NTA", "S"])

label = ["A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","NTA","A","NTA","A","NTA","S","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA","A","NTA"]
robus = [.488,.469,.485,.471,.473,.463,.480,.475,.473,.460,.467,.450,.426,.422,.365,.418,.001,.382,.357,.000,.512,.504,.520,.508,.520,.498,.505,.494,.505,.487,.497,.479,.484,.468,.470,.451,.452,.419,.416,.400,.370,.345]
accur = [.869,.896,.872,.891,.877,.896,.871,.890,.875,.895,.882,.899,.903,.908,.919,.888,.932,.917,.918,.953,.792,.856,.797,.863,.816,.872,.844,.883,.858,.888,.870,.892,.883,.897,.887,.905,.896,.914,.913,.919,.925,.929]
robus, accur, label = [np.array(x) for x in [robus, accur, label]]
#robus, accur, label = [x[19:] for x in [robus, accur, label]]

label_to_color = {"A": 0, "NTA": 1, "N": 2, "S": 3}
label_to_full_label = {"A": "PGD adv. training", "NTA": "Adaptive adv. training",
                       "N": "Noise training", "S": "Standard training"}

unique_colors = np.unique(label)


def get_series(lb):
    matches = (label == lb) & (accur >= 0)
    return accur[matches], robus[matches], lb


seriess = [get_series(c) for c in np.unique(label)]

zoom = True

lines = []

for a, r, lb in seriess:
    if zoom:
        a = a[r>0.3]
        r = r[r>0.3]
    if len(a) == 0:
        continue
    plt.scatter(r, a, label=label_to_full_label[lb])



plt.legend(loc="lower left")
plt.ylabel(r"$\mathrm{A}$")
plt.xlabel(r"$\mathrm{A}_{\mathrm{PGD}_{10}}$")
plt.title("DenseNet-121, CIFAR-10")
plt.show()
