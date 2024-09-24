import _context

import os
from pathlib import Path
import shutil

import torch
from tqdm import tqdm

import vidlu.utils.presentation.visualization as vis
import vidlu.transforms.image as vti
from vidlu.experiments import TrainingExperiment
import numpy as np


def untag(x):
    """Changes the type of an input from a vidlu.data.Domain subtype to torch.Tensor.

    The original object is unchanged.
    """
    return x.as_subclass(torch.Tensor)


def _perturb(attack, model, x, y, out):
    loss_mask = torch.ones_like(y[:, ...])
    pmodel = attack(model, x, y, loss_mask=loss_mask)
    pmodel.forward_arg_count = 4
    x_p, y_p, loss_mask_p = pmodel(x, y, loss_mask)
    _, out_po, _ = pmodel(x, out, loss_mask)
    oup_p = model(x_p)
    return x_p, y_p, oup_p, out_po


def generate_inputs(e: TrainingExperiment, n=8, dir="/tmp/semisup"):
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)
    os.makedirs(dir)

    trainer = e.trainer
    attack = trainer.attack
    model = e.model

    ds = e.data.test
    dl = trainer.data_loader_f(ds[:n], batch_size=1)

    examples = []
    for xy in tqdm(dl):
        x, y, *_ = trainer.prepare_batch(xy)
        x, y = untag(x), untag(y)
        out = model(untag(x))
        x_p, y_p, out_p, out_po = _perturb(attack=attack, model=model, x=x, y=y, out=out.argmax(1))
        examples.append(dict(x=x, y=y, x_p=x_p, y_p=y_p, out_po=out_po))

    torch.save(examples, dir / 'input_examples')
    return examples


def _generate_results(model, examples, dir="/tmp/semisup", suffix=""):
    if suffix != "":
        suffix = f"_{suffix}"
    dir = Path(dir)
    new = ["out", "out_p"]
    results = [dict(**item, out=model(item['x']).argmax(1), out_p=model(item['x_p']).argmax(1))
               for item in examples]
    results = [dict({k: v.view(v.shape[1:]).detach().cpu() for k, v in r.items()})
               for r in results]
    results = [
        dict({k: (vti.chw_to_hwc(v.clamp(0, 1)) if v.shape[0] == 3 else v).numpy() for k, v in
              r.items()})
        for r in results]

    colors = vis.normalize_colors(e.data.test.info.class_colors, insert_zeros=True)
    for i, r in enumerate(results):
        r = {k: v if len(v.shape) == 3 else vis.colorize_segmentation(v + 1, colors)
             for k, v in r.items()}
        r = {k: vti.numpy_to_pil((v * 255).astype(np.uint8)) for k, v in r.items()}
        for k, im in r.items():
            if k in new:
                path = dir / f"{i:04}_{k}{suffix}.png"
            else:
                path = dir / f"{i:04}_{k}.png"
            print(path)
            im.save(path)

    return results


def generate_results(e: TrainingExperiment, dir="/tmp/semisup", suffix=""):
    dir = Path(dir)
    model = e.model
    examples = torch.load(dir / 'input_examples')
    return _generate_results(model, examples, dir=dir, suffix=suffix)


def backenumerate(items):
    n = len(items)
    for i, x in enumerate(items):
        yield i - n, x


def print_latex_grid(n, dir):
    path = Path(dir)
    print(r"\begin{tabular}{@{}c@{\;}c@{\;}c@{\;}c@{}}")
    print(r"input & ground truth & simple-PhTPS & supervised \\")
    for i in range(n):
        for row, names in backenumerate([['x', 'y', 'out', 'out_sup'],
                                         ['x_p', 'y_p', 'out_p', 'out_p_sup']]):
            for col, name in backenumerate(names):
                print(f"\includegraphics[width=\imwidth]{{{path / f'{i:04}_{name}.png'}}}",
                      end='&\n' if col != -1 else '\\\\\n' if (i, row) != (n - 1, -1) else '\n')
    print("\end{tabular}")
