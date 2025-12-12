from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def export_feats(exp, split: str = 'train', feat_dir: str = 'FEATS/train', dtype: str = 'float16'):
    """
    Export per-segment features for the given split.

    Usage via run.py:
      python scripts/run.py test "...data..." "...model..." "...trainer..." -m "irap_gaim:export_feats,split='val',feat_dir='FEATS/val'"

    Args:
      exp: Vidlu experiment object passed by run.py test
      split: which split in exp.data to export ('train'|'val'|'test' or name prefix)
      feat_dir: output directory; files are saved as <segment_id>.npy
      dtype: float16|float32 for saving
    """
    assert hasattr(exp, 'trainer') and hasattr(exp, 'data')
    trainer = exp.trainer
    device = next(trainer.model.parameters()).device
    trainer.model.eval()

    # resolve dataset by split prefix
    dataset = None
    for name, ds in exp.data.items():
        if name.startswith(split):
            dataset = ds
            break
    if dataset is None:
        raise RuntimeError(f"No dataset starting with '{split}' found in experiment data.")

    out_dir = Path(feat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dl = trainer.get_data_loader(dataset, batch_size=trainer.eval_batch_size or 32, drop_last=False, shuffle=False)

    with torch.no_grad():
        for batch in dl:
            x = trainer.prepare_batch(batch)
            # Expect dict with 'rgb' and 'segment_id'
            # Ask model to return features if supported
            x_in = x[0] if isinstance(x, tuple) else x
            try:
                outputs, feats = trainer.model(x_in, return_features=True)
            except TypeError as e:
                raise RuntimeError("Model must support return_features=True for IRAP GAIM feature export.") from e
            feats = feats.detach().cpu().numpy()
            if dtype == 'float16':
                feats = feats.astype(np.float16)
            # Persist per sample
            segment_ids = batch['segment_id'] if isinstance(batch, dict) else batch[2]
            for i, sid in enumerate(segment_ids):
                sid_str = str(sid)
                np.save(out_dir / f"{sid_str}.npy", feats[i])

    return dict(saved_to=str(out_dir))


