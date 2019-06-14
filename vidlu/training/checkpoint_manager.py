import os
from pathlib import Path
import shutil
from argparse import Namespace
import warnings

import torch

from vidlu.utils.path import create_file_atomic


class CheckpointManager(object):
    """ Checkpoint manager can be used to periodically save objects to disk.
    Based on https://github.com/pytorch/ignite/ignite/handlers/checkpoint.py.

    Args:
        dir_path (str):
            Directory path where objects will be saved
        id (str):
            Prefix for the filenames to which objects will be saved.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be
            removed.
        overwrite_if_exists (bool, optional):
            If True, existing checkpoints with the same id will be deleted.
            Otherwise, exception will be raised if there are any files starting
            with `filename_prefix` in the directory 'dir_path'.
        resume (bool, optional):
            If True, `load_last` needs to be called in order to restore the last
            checkpoint. Then it will be assumed that training is continued. If
            `load_last` is not called before saved, an exception is raised.

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
    STATE_FILENAME = 'state.pth'
    PROGRESS_INFO_FILENAME = 'progress.info'
    EXPERIMENT_INFO_FILENAME = 'experiment.info'
    SUMMARY_FILENAME = 'log.txt'

    def __init__(self, checkpoints_dir, experiment_str, experiment_desc=None, n_saved=1,
                 resume=False,
                 remove_old=False):
        self.checkpoints_dir = checkpoints_dir
        self.exp_dir_path = Path(checkpoints_dir).expanduser() / experiment_str
        self.exp_dir_path.mkdir(parents=True, exist_ok=True)
        self.experiment_str = experiment_str
        self.experiment_desc = experiment_desc or dict()
        self._n_saved = n_saved
        self._index = 0
        self._required_resuming = resume

        def get_existing_checkpoints():
            paths = [str(p.stem) for p in self.exp_dir_path.iterdir()]
            indexes = map(self._name_to_index, paths)
            index_paths = sorted(zip(indexes, paths), key=lambda x: x[0])
            return [p for i, p in index_paths]

        self.saved = get_existing_checkpoints()

        if resume and remove_old:
            raise RuntimeError("`resume=True` is not allowed if `remove_old=True`.")
        if remove_old:
            self.remove_old_checkpoints(0)
        if resume and len(self.saved) == 0:
            raise RuntimeError("Cannot resume from checkpoint. Checkpoints not found.")
        if not resume and len(self.saved) > 0:
            raise RuntimeError(f"Files with ID {experiment_str} are already present in the "
                               + f"directory {checkpoints_dir}. If you want to use this ID anyway, pass"
                               + "either `remove_old=True` or `resume=True`. ")

    def save(self, state, summary=None):
        if self._required_resuming:
            raise RuntimeError("Cannot save state before it has been resumed. The `load_last`"
                               + " method needs to be called before saving to resume the training.")
        if len(state) == 0:
            raise RuntimeError("There are no objects to checkpoint in `state`.")

        self._index += 1

        name = self._index_to_name(self._index)
        path = self.exp_dir_path / name
        path.mkdir(parents=True, exist_ok=True)
        self._save(path / self.STATE_FILENAME, state)
        self._save(path / self.PROGRESS_INFO_FILENAME, Namespace(index=self._index))
        self._save(path / self.EXPERIMENT_INFO_FILENAME, self.experiment_desc)
        self._save(path / self.SUMMARY_FILENAME, summary)
        self.saved.append(name)

        self.remove_old_checkpoints()

    def load_last(self):
        path = self.exp_dir_path / self.saved[-1]
        self._index = torch.load(path / self.PROGRESS_INFO_FILENAME).index
        self.experiment_desc = torch.load(path / self.EXPERIMENT_INFO_FILENAME)
        state = torch.load(path / self.STATE_FILENAME)
        summary = torch.load(path / self.SUMMARY_FILENAME)
        self._required_resuming = False
        return state, summary

    def _save(self, path, obj):
        create_file_atomic(path=path, save_action=lambda file: torch.save(obj, file))

    def _index_to_name(self, index):
        return f'{index}'

    def _name_to_index(self, name):
        return int(name)

    def remove_old_checkpoints(self, n_saved=None):
        n_saved = self._n_saved if n_saved is None else n_saved
        while len(self.saved) > n_saved:
            path = self.exp_dir_path / self.saved.pop(0)
            try:
                shutil.rmtree(path)
            except FileNotFoundError as ex:
                warnings.warn(f"Old checkpoint {path} is already deleted.")
