import dataclasses as dc
from pathlib import Path
import shutil
import warnings
import typing as T
import logging

from vidlu.utils.path import create_file_atomic
from vidlu.utils.func import params
from vidlu.utils.loadsave import TorchLoadSave, JsonLoadSave, TextLoadSave


class _Smallest(float):  # float inheritance needed storing as JSON
    def __new__(cls):
        return float.__new__(cls, "-inf")

    def __lt__(self, other):
        return not isinstance(other, _Smallest)

    def __gt__(self, other):
        return False


smallest = _Smallest()


class Files:
    extracted_state = (None, TorchLoadSave)
    state = ('training_state.pth', TorchLoadSave)
    progress_info = ('progress.info', TorchLoadSave)
    info = ('experiment.info', TorchLoadSave)
    summary = ('summary.p', TorchLoadSave)
    perf = ('perf.json', JsonLoadSave)
    log = ('log.txt', TextLoadSave)


@dc.dataclass
class Checkpoint:  # TODO
    info: T.Mapping
    state: T.Mapping  # special
    summary: T.Mapping
    progress_info: T.Mapping

    perf: object = smallest
    log: str = ""

    def save(self, path):
        separately_saved_state_parts = self.info['separately_saved_state_parts']
        extracted_state_parts = {f"{k}_state": self.state[k] for k in separately_saved_state_parts}
        other_state = {k: v for k, v in self.state.items() if k not in separately_saved_state_parts}
        stuff = dict(**{**{k: getattr(self, k) for k in self.__annotations__},
                        'state': other_state}, **extracted_state_parts)
        try:
            for k, v in stuff.items():
                name, file_interface = getattr(Files, k, None) or (f"{k}.pth", TorchLoadSave)
                create_file_atomic(path=path / name, mode=file_interface.write_mode,
                                   write_action=lambda f: file_interface.save(v, f))
        except Exception as e:
            shutil.rmtree(path)
            raise

    @classmethod
    def load(cls, path, map_location=None):
        fields = {k: (getattr(cls, k) if k in ("perf", "log") else
                      Checkpoint._load(path, k, map_location=map_location))
                  for k in cls.__annotations__}
        for k in fields['info']['separately_saved_state_parts'] \
                if isinstance(fields['info'], T.Mapping) else ['model']:  # TODO: remove ['model']
            fields['state'][k] = cls._load(path, f"{k}_state", map_location=map_location)
        return Checkpoint(**fields)

    @staticmethod
    def _load(path, k, map_location=None):
        name, fi = getattr(Files, k, None) or (f"{k}.pth", TorchLoadSave)
        path = path / name
        return fi.load(path, map_location=map_location) if "map_location" in params(fi.load) \
            else fi.load(path)


Mode = T.Literal['restart', 'resume', 'resume_or_start', 'start']


class CheckpointManager(object):
    """Checkpoint manager can be used to periodically save objects to disk.
    Based on https://github.com/pytorch/ignite/ignite/handlers/checkpoint.py.

    Args:
        checkpoints_dir (str):
            Directory path where objects will be saved
        experiment_name (str):
            Prefix of the file paths to which objects will be saved.
        n_last_kept (int, optional):
            Number of objects that should be kept on disk. Older files will be
            removed.
        resume (bool, optional):
            If True, `load_last` needs to be called in order to restore the last
            checkpoint and continue. If `load_last` is not called before
            saving/updating, an exception is raised.
        reset (bool, optional):
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

    def __init__(self, checkpoints_dir, experiment_name: str, info: T.Mapping = None,
                 n_last_kept=1, n_best_kept=0, mode: Mode = 'start',
                 separately_saved_state_parts: T.Sequence[str] = (), perf_func=lambda s: smallest,
                 log_func=lambda s: "", name_suffix_func=lambda s: ""):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._logger.addHandler(logging.NullHandler())

        self.checkpoints_dir = checkpoints_dir
        self.experiment_dir = Path(checkpoints_dir).expanduser() / experiment_name
        self.info = info or dict()
        self.n_recent_kept, self.n_best_kept = n_last_kept, n_best_kept
        self.separately_saved_state_parts = separately_saved_state_parts
        self.perf_func, self.log_func, self.name_suffix_func = perf_func, log_func, name_suffix_func

        self.index = 0
        self.sync()
        self.resuming_required = mode != "restart" and len(self.saved) > 0

        if mode == "restart":
            self.restart()
        elif mode == "resume":
            if not self.resuming_required:
                raise RuntimeError(f"Cannot resume from checkpoint. Checkpoints not found in"
                                   + f" {self.experiment_dir}.")
        elif mode == "start":
            if self.resuming_required:
                raise RuntimeError(f"{experiment_name} is already present in {checkpoints_dir}. If"
                                   + " you want to use this ID anyway, pass `mode='resume'`.")
        elif mode != "resume_or_start":
            raise ValueError(f"Argument {mode=} does not match {Mode}.")

    def restart(self):
        self.remove_old_checkpoints(0, 0)  # does not remove checkpoints when called from __init__
        self.index = 0
        self.resuming_required = False

    def sync(self):
        def get_existing_checkpoints():
            if not self.experiment_dir.exists():
                return []
            return sorted([p.name for p in self.experiment_dir.iterdir()],
                          key=lambda p: int(p.split("_")[0]))

        self.saved = get_existing_checkpoints()
        self.id_to_perf = dict()
        for id in list(self.saved):
            try:
                self.id_to_perf[id] = self.perf_func(
                    Checkpoint.load(self.experiment_dir / id).summary)
            except EOFError as e:
                warnings.warn(f"Checkpoint loading error:\n{e}")
                self.saved.remove(id)

    def save(self, state, summary=None):
        if self.resuming_required:
            raise RuntimeError("Cannot save state before resuming. The `load_last` method"
                               + " needs to be called before saving to resume the training.")
        if len(state) == 0:
            raise RuntimeError("There are no objects to checkpoint in `state`.")

        self.index += 1

        cp = Checkpoint(state=state, summary=summary, progress_info=dict(index=self.index),
                        info=dict(info=self.info,
                                  separately_saved_state_parts=self.separately_saved_state_parts),
                        perf=self.perf_func(summary), log=self.log_func(summary))
        name = f"{self.index}" + (
            "" if (suff := self.name_suffix_func(summary)) == "" else f"_{suff}")
        path = self.experiment_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Saving checkpoint {name} in {self.experiment_dir}.")
        cp.save(path)
        self.saved.append(name)
        self.id_to_perf[name] = cp.perf
        self.remove_old_checkpoints()

    @property
    def last_checkpoint_path(self):
        return self.experiment_dir / self.saved[-1] if len(self.saved) > 0 else None

    @property
    def best_checkpoint_path(self):
        return self.experiment_dir / max(self.saved, key=self.id_to_perf.__getitem__) \
            if len(self.saved) > 0 else None

    def load_last(self, map_location=None):
        return self._load(self.last_checkpoint_path, map_location=map_location)

    def load_best(self, map_location=None):
        return self._load(self.best_checkpoint_path, map_location=map_location)

    def _load(self, path, map_location=None):
        self._logger.info(f"Loading checkpoint {path.name}.")
        cp = Checkpoint.load(path, map_location=map_location)
        self.index = (pi if isinstance(pi := cp.progress_info, dict) else pi.__dict__)['index']
        self.info = cp.info
        self.resuming_required = False
        return cp.state, cp.summary

    def remove_old_checkpoints(self, n_recent_kept=None, n_best_kept=None):
        n_recent_kept = self.n_recent_kept if n_recent_kept is None else n_recent_kept
        n_best_kept = self.n_best_kept if n_best_kept is None else n_best_kept
        best = set(() if n_best_kept == 0 else
                   sorted(self.saved, key=self.id_to_perf.__getitem__)[-n_best_kept:])
        olds = self.saved[:-n_recent_kept] if n_recent_kept > 0 else self.saved[:]
        removed = [x for x in olds if x not in best]
        for p in removed:
            path = self.experiment_dir / p
            self.saved.remove(p)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                warnings.warn(f"Old checkpoint {path} is already deleted.")
        self.id_to_perf = {k: self.id_to_perf[k] for k in self.saved}
        if len(self.saved) == 0 and self.experiment_dir.exists():
            shutil.rmtree(self.experiment_dir)
