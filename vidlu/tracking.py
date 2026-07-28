"""Concrete experiment-tracking backends for the `MetricTracker` protocol
defined in `vidlu.experiments`."""

import os
import typing as T
from pathlib import Path

import numpy as np

from vidlu.utils import tree


def flatten_scalars(metrics: T.Mapping[str, T.Any]) -> dict:
    """Flattens nested dicts to dotted keys and keeps only scalar values.

    Numpy scalars and 0-d arrays are converted to Python scalars. Arrays with
    `ndim >= 1` (e.g. confusion matrices) and other non-scalar values are
    dropped.
    """
    out = {}
    for path, v in tree.flatten(metrics, tree_type=T.Mapping):
        key = ".".join(map(str, path))
        if isinstance(v, (bool, int, float, np.floating, np.integer)):
            out[key] = v.item() if isinstance(v, (np.floating, np.integer)) else v
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[key] = v.item()
    return out


class WandbTracker:
    """A `MetricTracker` backed by Weights & Biases.

    The run id is persisted to `run_id_path` so that a resumed experiment
    continues the same W&B run. The W&B entity/project/mode are configured via
    the standard `WANDB_ENTITY`/`WANDB_PROJECT`/`WANDB_MODE` environment
    variables (the project defaults to "vidlu").

    Args:
        run_name: Display name of the run (e.g. the vidlu experiment name).
        config: Serializable experiment configuration stored with the run.
        run_id_path: File in which the W&B run id is persisted.
        resume: Whether to continue the run whose id is stored at
            `run_id_path`, if it exists.
    """

    def __init__(self, run_name: str, config: T.Mapping[str, T.Any], run_id_path: Path,
                 resume: bool):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                'The "wandb" tracker requires the wandb package: pip install wandb.') from e
        self._wandb = wandb
        run_id = (run_id_path.read_text().strip()
                  if resume and run_id_path.exists() else None) or None
        self._run = wandb.init(project=os.environ.get("WANDB_PROJECT", "vidlu"),
                               name=run_name, id=run_id,
                               resume="allow" if run_id else None, config=dict(config))
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(self._run.id)
        # Start the monotonic clamp from the resumed run's last step so that
        # e.g. the initial evaluation of a resumed experiment is not logged at
        # step 0.
        self._last_step = self._run.step or 0

    def _clamp_step(self, step: int) -> int:
        # W&B requires non-decreasing steps; clamping absorbs edge cases such as
        # the initial evaluation before any training step.
        self._last_step = max(int(step), self._last_step)
        return self._last_step

    def _log(self, payload: T.Mapping[str, T.Any], step: int, split: str | None) -> None:
        """Applies the split prefix, clamps the step, and logs a non-empty payload."""
        if split is not None:
            payload = {f"{split}/{k}": v for k, v in payload.items()}
        step = self._clamp_step(step)
        if payload:
            self._wandb.log(payload, step=step)

    def log_scalars(self, metrics: T.Mapping[str, T.Any], step: int,
                    split: str | None = None) -> None:
        self._log(flatten_scalars(metrics), step, split)

    def log_images(self, images: T.Mapping[str, T.Any], step: int,
                   split: str | None = None) -> None:
        """Logs images (tensors, ndarrays, PIL images, or `wandb.Image`s)."""
        self._log({k: v if isinstance(v, self._wandb.Image) else self._wandb.Image(v)
                   for k, v in images.items()}, step, split)

    def finish(self, exit_code: int = 0) -> None:
        self._wandb.finish(exit_code=exit_code)
