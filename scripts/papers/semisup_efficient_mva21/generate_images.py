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



def _perturb(attack, model, x, target):
    loss_mask = torch.ones_like(target[:, 0, ...])
    pmodel = attack(model, x, target, loss_mask=loss_mask)
    x_p, target_p, loss_mask_p = pmodel(x, target, loss_mask)
    out_p = model(x_p)
    return x_p, target_p, out_p


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
        x, y = trainer.prepare_batch(xy)
        x_p, y_p, out_p = _perturb(attack=attack, model=model, x=x, target=y)
        examples.append(dict(x=x, y=y, x_p=x_p, y_p=y_p))

    torch.save(examples, dir / 'input_examples')
    return examples


def generate_results(e: TrainingExperiment, dir="/tmp/semisup", suffix=""):
    if suffix != "":
        suffix = f"_{suffix}"
    dir = Path(dir)
    model = e.model
    examples = torch.load(dir / 'input_examples')
    results = [dict(**item, out=model(item['x']).argmax(1), out_p=model(item['x_p']).argmax(1))
               for item in examples]
    results = [dict({k: v.view(v.shape[1:]).detach().cpu() for k, v in r.items()})
               for r in results]
    results = [
        dict({k: (vti.chw_to_hwc(v.clamp(0, 1)) if v.shape[0] == 3 else v).numpy() for k, v in
              r.items()})
        for r in results]

    colors = vis.normalize_colors(e.data.test.info.class_colors)
    for i, r in enumerate(results):
        r = {k: v if len(v.shape) == 3 else vis.colorize_segmentation(v + 1, colors)
             for k, v in r.items()}
        r = {k: vti.numpy_to_pil((v * 255).astype(np.uint8)) for k, v in r.items()}
        for k, im in r.items():
            if k.startswith('out'):
                path = dir / f"{i:04}_{k}.png"
            else:
                path = dir / f"{i:04}_{k}{suffix}.png"
            print(path)
            im.save(path)

    return results


def latex_grid(n, dir):
    path = Path(dir)
    for i in range(n):
        for names in [['x', 'y', 'out', 'out_sup'], ['x_p', 'y_p', 'out_p', 'out_p_sup']]:
            for name in names:
                print(f"\includegraphics[width=\imwidth]{{{path / f'{i:04}_{name}.png'}}}",
                      end='\,')
            print(r"\\")
