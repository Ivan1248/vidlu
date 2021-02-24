import argparse

import torch
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import _context
from vidlu.utils.collections import NameDict
from vidlu.utils.func import tryable
from vidlu.utils.presentation.visualization import plot_curves
import parse

"""
/home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-15-0805/Model.log
/home/igrubisic/projects/dl-uncertainty/data/nets/cifar-trainval/dn-100-12-e300/2018-05-18-0725/Model.log
/home/igrubisic/projects/dl-uncertainty/data/nets/cityscapes-train/dn-121-32-pretrained-e30/2018-05-18-0010/Model.log
"""

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--include', type=str, default=None)
parser.add_argument('--exclude', type=str, default="")
args = parser.parse_args()

with open(args.path, 'rb') as summary_file:
    data = NameDict(torch.load(summary_file))

batches_per_epoch = []
epoch_lr = [NameDict(epoch=0, lr=0)]
train_evals = []
val_evals = []

r"""
scanner_action = [
    (FormatScanner(
        "[([:\d]+)] Epoch {epoch:(\d+)}/(\d+) (\(){batches:(\d+)} batches(\)), lr=[([^\]])]",
        full_match=False),
     lambda batches: batches_per_epoch.append(batches)),
    (FormatScanner("[([:\d]+)] {epoch:(\d+)}.{last_batch(\d+)}: {evaluation:(.*)}",
                   full_match=True),
     lambda epoch, last_batch, evaluation: train_evals.append(
         Namespace(epoch=epoch, last_batch=last_batch, evals=eval(f"dict({evaluation})")))),
    (FormatScanner("[([:\d]+)] val: {evaluation:(.*)}",
                   full_match=True),
     lambda evaluation: val_evals.append(eval(f"dict({evaluation})"))),
]
"""

scanner_action = [
    # batches_per_epoch, epoch_lr
    (lambda x: parse.parse("{} Epoch {epoch:d}/{:d} ({batches:d} batches, lr={lr:e}{}",
                           x).named,
     lambda epoch, batches, lr: (print(batches), (batches_per_epoch.append(batches),
                                                  epoch_lr.append(NameDict(epoch=epoch, lr=lr))))),
    # train_evals
    (lambda x: parse.parse("{}/s, {epoch:d}.{last_batch:d}: {evaluation}", x).named,
     lambda epoch, last_batch, evaluation: train_evals.append(
         NameDict(epoch=epoch, last_batch=last_batch, evals=eval(f"dict({evaluation})")))),
    # val_evals
    (lambda x: parse.parse("{} val: {}/s, {evaluation}", x).named,
     lambda evaluation: val_evals.append(
         eval(f"NameDict(evals=NameDict({evaluation}), epoch={epoch_lr[-1].epoch})"))),
    # # error
    # (lambda x: parse.parse("{}", x),
    #  lambda line: print(f"No parsing rule defined for line\n{line}")),
]

includes = None if args.include is None else args.include.split(",")
excludes = args.exclude.split(",")


def name_filter(name):
    if includes is None:
        return not any(x.startswith(name) for x in excludes)
    return any(x.startswith(name) for x in includes)


def parse_line(line):
    for scanner, action in scanner_action:
        # data = scanner(line)
        data = tryable(scanner, None)(line)
        if data:
            action(**data)
            return True
    return False


for line in data.logger['lines']:
    print(line)
    if not parse_line(line):
        print(f"Unparsed: {line}")

batches_per_epoch = batches_per_epoch[0]

epoch_count = val_evals[-1].epoch
train_curve_x = [(x.epoch - 1) + x.last_batch / batches_per_epoch for x in train_evals]
train_curve_ys = dict(zip(train_evals[0].evals.keys() if len(train_evals) > 0 else (),
                          (list(zip(*[e.evals.values() for e in train_evals])))))
train_curves = NameDict({f"{k}_train": (train_curve_x, y) for k, y in train_curve_ys.items()})

val_curve_x = [x.epoch for x in val_evals]
val_curve_ys = dict(zip(val_evals[0].evals.keys(),
                        (list(zip(*[e.evals.values() for e in val_evals])))))
val_curves = NameDict({k: (val_curve_x, y) for k, y in val_curve_ys.items()})

curves = {**train_curves, **val_curves}
curves = {k: v for k, v in curves.items() if name_filter(k)}

plot_curves(curves)
plt.show()