import os
import dataclasses as dc
from pathlib import Path
import shutil
import warnings

import torch

from vidlu.utils.path import create_file_atomic


class FileNames:
    MODEL_STATE = 'model_state.pth'
    TRAINING_STATE = 'training_state.pth'
    PROGRESS_INFO = 'progress.info'
    EXPERIMENT_DESC = 'experiment.info'
    SUMMARY = 'summary.p'


@dc.dataclass
class Checkpoint:  # TODO
    model_state: dict
    training_state: dict
    progress_info: dict
    experiment_desc: dict
    summary: dict

    def save(self, path):
        for k, v in vars(self).items():
            self._save(path / getattr(FileNames, k.upper()), v)

    @classmethod
    def load(cls, path, map_location=None):
        return cls(**{k: torch.load(path / getattr(FileNames, k.upper()), map_location=map_location)
                      for k, v in cls.__annotations__.items()})

    @staticmethod
    def _save(path, obj):
        create_file_atomic(path=path, save_action=lambda file: torch.save(obj, file))


class CheckpointManager(object):
    """ Checkpoint manager can be used to periodically save objects to disk.
    Based on https://github.com/pytorch/ignite/ignite/handlers/checkpoint.py.

    Args:
        checkpoints_dir (str):
            Directory path where objects will be saved
        experiment_name (str):
            Prefix of the file paths to which objects will be saved.
        n_recent_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be
            removed.
        resume (bool, optional):
            If True, `load_last` needs to be called in order to restore the last
            checkpoint and continue. If `load_last` is not called before
            saving/updating, an exception is raised.
        remove_old (bool, optional):
            If True, existing checkpoints with the same checkpoints_dir and
            experiment_name will be deleted.

    Notes:
        These names are used to specify filenames for saved objects. Each
        saved state directory a name with the following format: `{id}_{index}`,
        where `id` is the argument passed to the constructor, and `index` is the
        index of the saved object, which is incremented by 1 with every call to
        `save`. The directory contains the following files: 'state.pth' - the
        state object passed to `save`, 'checkpoint.info' - CheckpointManager
        state, and, optionally, 'log.txt' - the log (lines) provided as the
        second argument to `save`.

    Examples:
        >>> import os
        >>> from vidlu.training.checkpoint_manager import CheckpointManager
        >>> from torch import nn
        >>> model = nn.Linear(3, 3)
        >>> cpman = CheckpointManager('/tmp/states', 'lin33', n_saved=2, overwrite_if_exists=True)
        >>> cpman.save(model.state_dict(), summary=dict(lines=['baz']))
        >>> cpman.save(model.state_dict(), summary=dict(lines=['baz', 'bap']))
        >>> state, logger_state = cpman.load_last()
        >>> model.load_state_dict(state)
        >>> assert logger_state == dict(lines=['baz', 'bap'])
        >>> os.listdir('/tmp/states')
        ['lin33_1', 'lin33_2']
    """

    def __init__(self, checkpoints_dir, experiment_name, experiment_desc=None, n_recent_saved=1,
                 n_best_saved=0, resume=False, remove_old=False, perf_measure=lambda cp: 0):
        self.checkpoints_dir = checkpoints_dir
        self.experiment_dir = Path(checkpoints_dir).expanduser() / experiment_name
        self.experiment_str = experiment_name
        self.experiment_desc = experiment_desc or dict()
        self.n_recent_saved = n_recent_saved
        self.n_best_saved = n_best_saved
        self._index = 0
        self._required_resuming = resume

        def get_existing_checkpoints():
            if not self.experiment_dir.exists():
                return []
            paths = [str(p.stem) for p in self.experiment_dir.iterdir()]
            indexes = map(self._name_to_index, paths)
            index_paths = sorted(zip(indexes, paths), key=lambda x: x[0])
            return [p for i, p in index_paths]

        self.saved = get_existing_checkpoints()
        self.path_to_performance = {p: perf_measure(Checkpoint.load(self.experiment_dir / p))
                                    for p in self.saved}

        if resume and remove_old:
            raise RuntimeError("`resume=True` is not allowed if `remove_old=True`.")
        if remove_old:
            self.remove_old_checkpoints(0)
        if resume and len(self.saved) == 0:
            raise RuntimeError(f"Cannot resume from checkpoint. Checkpoints not found in"
                               + f" {self.experiment_dir}.")
        if not resume and len(self.saved) > 0:
            raise RuntimeError(f"Files with ID {experiment_name} are already present in the "
                               + f"directory {checkpoints_dir}. If you want to use this ID anyway, "
                               + "pass either `remove_old=True` or `resume=True`. ")

    @property
    def last_checkpoint_path(self):
        return self.experiment_dir / self.saved[-1]

    @property
    def best_checkpoint_path(self):
        return self.experiment_dir / max(self.saved, key=self.path_to_performance.__getitem__)

    def save(self, state, summary=None):
        if self._required_resuming:
            raise RuntimeError("Cannot save state before resuming. The `load_last` method"
                               + " needs to be called before saving to resume the training.")
        if len(state) == 0:
            raise RuntimeError("There are no objects to checkpoint in `state`.")

        self._index += 1

        name = self._index_to_name(self._index)
        path = self.experiment_dir / name
        path.mkdir(parents=True, exist_ok=True)

        Checkpoint(model_state=state['model'],
                   training_state={k: v for k, v in state.items() if k != 'model'},
                   progress_info=dict(index=self._index),
                   experiment_desc=self.experiment_desc,
                   summary=summary).save(path)

        self.saved.append(name)

        self.remove_old_checkpoints()

    def load_last(self, map_location=None):
        path = self.last_checkpoint_path

        cp = Checkpoint.load(path, map_location=map_location)
        self._index = (pi if isinstance(pi := cp.progress_info, dict) else pi.__dict__)['index']
        self.experiment_desc = cp.experiment_desc
        state = cp.training_state
        state['model'] = cp.model_state
        summary = cp.summary

        self._required_resuming = False
        return state, summary

    @staticmethod
    def _save(path, obj):
        create_file_atomic(path=path, save_action=lambda file: torch.save(obj, file))

    @staticmethod
    def _index_to_name(index):
        return f'{index}'

    @staticmethod
    def _name_to_index(name):
        return int(name)

    def remove_old_checkpoints(self, n_recent_saved=None, n_best_saved=None):
        n_recent_saved = self.n_recent_saved if n_recent_saved is None else n_recent_saved
        n_best_saved = self.n_best_saved if n_best_saved is None else n_best_saved
        best = set(() if n_best_saved == 0 else
                   sorted(self.saved, key=self.path_to_performance.__getitem__)[-n_best_saved:])
        for p in self.saved[:-n_recent_saved]:
            path = self.experiment_dir / p
            if path in best:
                continue
            self.saved.remove(p)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                warnings.warn(f"Old checkpoint {path} is already deleted.")
