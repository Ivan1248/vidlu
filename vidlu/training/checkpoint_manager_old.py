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
        >>> cpman.save(model.state_dict(), log=['baz'])
        >>> cpman.save(model.state_dict(), log=['baz', 'bap'])
        >>> state, log = cpman.load_last()
        >>> model.load_state_dict(state)
        >>> assert log == ['baz', 'bap']
        >>> os.listdir('/tmp/states')
        ['lin33_1', 'lin33_2']
    """
    STATE_FILENAME = 'state.pth'
    INFO_FILENAME = 'checkpoint.info'
    LOG_FILENAME = 'log.txt'

    def __init__(self, dir_path, id, n_saved=1, overwrite_if_exists=False, resume=False):
        self._dir_path = Path(dir_path).expanduser()
        self._dir_path.mkdir(parents=True, exist_ok=True)
        self._id = id
        self._n_saved = n_saved
        self._index = 0
        self._required_resuming = resume

        def get_existing_checkpoints():
            paths = [str(p) for p in self._dir_path.iterdir() if p.name.startswith(self._id)]
            indexes = map(self._name_to_index, paths)
            index_paths = sorted(zip(indexes, paths), key=lambda x: x[0])
            return [p for i, p in index_paths]

        self.saved = get_existing_checkpoints()

        if overwrite_if_exists and resume:
            raise RuntimeError("`resume=True` is not allowed if `overwrite_if_exists=True`.")
        if overwrite_if_exists:
            self.remove_old_checkpoints(0)
        elif not resume and len(self.saved) > 0:
            raise RuntimeError(f"Files with ID {id} are already present in the directory "
                               + f"{dir_path}. If you want to use this ID anyway, pass either"
                               + "`overwrite_if_exists=True` or `resume=True`. ")
        if resume and len(self.saved) == 0:
            raise RuntimeError("Cannot resume from checkpoint. No checkpoint found.")

    def save(self, state, log=None):
        if self._required_resuming:
            raise RuntimeError("Cannot save state before it has been resumed. `load_last` needs"
                               + " to be called  before saving for the training to be considered"
                               + " resumed. ")
        if len(state) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._index += 1

        name = self._index_to_name(self._index)
        path = self._dir_path / name
        path.mkdir()
        self._save(path / self.STATE_FILENAME, state)
        self._save(path / self.INFO_FILENAME, Namespace(index=self._index))
        if log is not None:
            (path / self.LOG_FILENAME).write_text('\n'.join(log))
        self.saved.append(path)

        self.remove_old_checkpoints()

    def load_last(self):
        path = self._dir_path / self.saved[-1]
        self._index = torch.load(path / self.INFO_FILENAME).index
        state = torch.load(path / self.STATE_FILENAME)
        log_path = path / self.LOG_FILENAME
        log = log_path.read_text().split('\n') if log_path.exists() else []
        self._required_resuming = False
        return state, log

    def _save(self, path, obj):
        create_file_atomic(path=path,
                           save_action=lambda file: torch.save(obj, file))

    def _index_to_name(self, index):
        return f'{self._id}_{index}'

    def _name_to_index(self, name):
        return int(name[name.rindex('_') + 1:])

    def remove_old_checkpoints(self, n_saved=None):
        n_saved = self._n_saved if n_saved is None else n_saved
        while len(self.saved) > n_saved:
            path = self.saved.pop(0)
            try:
                shutil.rmtree(path)
            except FileNotFoundError as ex:
                warnings.warn(f"Old checkpoint {path} is already deleted.")
