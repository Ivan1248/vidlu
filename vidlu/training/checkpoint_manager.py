import dataclasses as dc
import os
from pathlib import Path
import shutil
import warnings
import typing as T
import logging
import math

import torch

from vidlu.utils.path import create_file_atomic
from vidlu.utils.func import params
from vidlu.utils.storage import TorchLoadSave, JsonLoadSave, TextLoadSave


class _SmallestFloat(float):  # float inheritance needed storing as JSON
    def __new__(cls):
        return float.__new__(cls, "-inf")

    def __lt__(self, other):
        return not isinstance(other, _SmallestFloat)

    def __gt__(self, other):
        return False


smallest = _SmallestFloat()


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

    perf: float = smallest
    log: str = ""

    def save(self, path):
        separately_saved_state_parts = self.info['separately_saved_state_parts']
        extracted_state_parts = {f"{k}_state": self.state[k] for k in separately_saved_state_parts}
        other_state = {k: v for k, v in self.state.items() if k not in separately_saved_state_parts}

        fields = {k: getattr(self, k) for k in self.__annotations__}
        stuff = dict(**{**fields, 'state': other_state}, **extracted_state_parts)

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
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
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
        try:
            return fi.load(path, map_location=map_location) if "map_location" in params(fi.load) \
                else fi.load(path)
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint at {path.parent}. Error {e}.")


ModeArg = T.Literal['restart', 'resume', 'resume_or_start', 'start']


class CheckpointManager(object):
    """Checkpoint manager can be used to periodically save algorithm states and summaries to disk.
    
    Based on https://github.com/pytorch/ignite/ignite/handlers/checkpoint.py.

    The main methods are `save`, `load_last` and `load_best` which receive or return a state
    dictionary.

    Args:
        checkpoints_root (str | os.PathLike): Directory path where objects will be saved
        experiment_name (str): Prefix of the name of the directory corresponding to a checkpoint.
        experiment_info (object, optional): Arbitrary fixed data about the experiment that is saved
            to the disk for all checkpoints.
        n_recent_kept (int, optional): Maximum number of most recent checkpoints to be  kept on
            disk. Older checkpoints are removed. Setting the value to `inf` keeps all checkpoints.
            Default: 1.
        n_best_kept (int, optional): Maximum number of best checkpoints kept on disk. Default: 0.
        start_mode (Literal['restart', 'resume', 'resume_or_start', 'start'], optional): Controls
            whether an exception should be raised depending on whether checkpoints of the experiment
            already exist and whether the user will load a checkpoint before saving a new one:
            - "restart" removes all existing checkpoints on initialization,
            - "resume" expects that the user will first load an existing checkpoint,
            - "resume_or_start" expects that the user will first load a checkpoint if any exist,
            - "start" expects that there are no existing checkpoints.
            Default; 'start'.
        separately_saved_state_parts (Sequence[str], optional): Keys of entries in the state
            dictionary that are to be saved as separate files in checkpoint directories. This does
            not affect the external behavior of the `save` and `load_*` methods. Default: `()`.
        perf_func (Callable[[Mapping], float], optional): A function that receives the second
            argument of the `save` method and returns a `float` that can be used to compare
            checkpoints (greater is better). Default: constant "smallest float" function.
        log_func (Callable[[Mapping], str], optional): A function that receives the second argument
            of the `save` method and returns a string that should be printed when an expeiment is
            resumed. Default: `lambda s: ""`.
        name_suffix_func (Callable[[Mapping], str], optional): A function that receives the second
            argument of the `save` method and returns a suffix that is appended to the name of the
            checkpoint directory `lambda s: ""`.
        resume (bool, optional):
            If True, `load_last` needs to be called in order to restore the last checkpoint and
            continue. If `load_last` is not called before saving/updating, an exception is raised.
        reset (bool, optional):
            If True, existing checkpoints with the same checkpoints_dir and experiment_name will be
            deleted.

    Notes:
        A checkpoint is saved as a directory under a path with the format:
            `{checkpoints_dir}/{experiment_name}/{checkpoint_index}{name_suffix}`,
        where:
        - `checkpoints_dir` and `experiment_name` are constructor arguments.
        - `checkpoint_index` starts from 0 and increases each time a checkpoint is saved.
        - `name_suffix` is an optional suffix that depends on `name_suffix_func` and the `summary`
        argument of the `save` method.
        Each checkpoint directory contains files that correspond to `CheckpointManager` fields and
        extracted state dictionary entries as defined by `separately_saved_state_parts`.

    Examples:
        >>> import os
        >>> from vidlu.training.checkpoint_manager import CheckpointManager
        >>> from torch import nn
        >>> checkpoints_root = '~/data/experiments/states'
        >>> model = nn.Linear(3, 3)
        >>> cpman = CheckpointManager(
        ...     checkpoints_root=checkpoints_root,  # directory that contains experiment directories
        ...     experiment_name='lin33',  # name of experiment directory, which contains checkpoints
        ...     n_recent_kept=2,  # store only the 2 most recent checkpoint
        ...     start_mode='resume_or_start',  # resume if any checkpoints exist
        ...     perf_func=lambda s: s.get('perf', 0),  # extract performance from summary
        ...     log_func=lambda s: s.get('log', ""),
        ...     name_suffix_func=lambda s: f"{s['epoch']}_{s['perf']:.2f}")
        >>> if cpman.resuming_required:
        >>>     state, summary = cpman.load_last(map_location='cuda:0')
        >>>     print(summary['log'])
        >>> cpman.save(model.state_dict(), summary=dict(perf=0.11, log='Starting', epoch=0))
        >>> cpman.save(model.state_dict(), summary=dict(perf=0.25, epoch=10, log='Starting\\nBaz'))
        >>> state, summary = cpman.load_last(map_location='cuda:0')
        >>> model.load_state_dict(state)
        >>> assert summary['log'] == 'Starting\\nBaz'
        >>> os.listdir(f'{checkpoints_dir}/lin33')
        ['0_0_0.11', '1_10_0.25']
    """

    def __init__(self, checkpoints_root: str | os.PathLike, experiment_name: str,
                 experiment_info: T.Mapping = None,
                 n_recent_kept: T.Union[int, T.Literal[math.inf]] = 1, n_best_kept: int = 0,
                 start_mode: T.Optional[ModeArg] = 'start',
                 separately_saved_state_parts: T.Sequence[str] = (),
                 state_loaded: bool = False,
                 perf_func: T.Callable[[T.Mapping], float] = lambda s: smallest,
                 log_func: T.Callable[[T.Mapping], str] = lambda s: "",
                 name_suffix_func=lambda s: ""):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._logger.addHandler(logging.NullHandler())

        self.checkpoints_root = Path(checkpoints_root).expanduser()
        self.experiment_name = experiment_name
        self.info = experiment_info or dict()
        self.n_recent_kept, self.n_best_kept = n_recent_kept, n_best_kept
        self.separately_saved_state_parts = separately_saved_state_parts
        self.perf_func, self.log_func, self.name_suffix_func = perf_func, log_func, name_suffix_func

        self.last_index = -1
        self.sync()
        self.resuming_required = not state_loaded and start_mode != "restart" and len(
            self.saved) > 0

        if start_mode == "restart":
            self.restart()
        elif start_mode == "resume":
            if not self.resuming_required:
                raise RuntimeError(f"Cannot resume from checkpoint. Checkpoints not found in"
                                   + f" {self.experiment_dir}.")
        elif start_mode == "start":
            if self.resuming_required:
                raise RuntimeError(f"{experiment_name} is already present in {checkpoints_root}."
                                   + " You can resume or restart.")
        elif start_mode != "resume_or_start":
            raise ValueError(f"Argument {start_mode=} does not match type {ModeArg}.")

    @property
    def experiment_dir(self):
        return self.checkpoints_root / self.experiment_name

    def __len__(self):
        return len(self.saved)

    def restart(self):
        self.remove_old_checkpoints(0, 0)  # does not remove checkpoints when called from __init__
        self.last_index = -1
        self.resuming_required = False

    def sync(self):
        def get_existing_checkpoints():
            if not self.experiment_dir.exists():
                return []
            return sorted([p.name for p in self.experiment_dir.iterdir() if not p.is_file()],
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

    def save(self, state, summary=None, index=None):
        if self.resuming_required:
            raise RuntimeError("`load_last` or some other `load*` method"
                               + " must be called before saving to resume the training.")
        if len(state) == 0:
            raise RuntimeError("There are no objects to checkpoint in `state`.")

        self.last_index = self.last_index + 1 if index is None else index

        cp = Checkpoint(state=state, summary=summary, progress_info=dict(index=self.last_index),
                        info=dict(info=self.info,
                                  separately_saved_state_parts=self.separately_saved_state_parts),
                        perf=self.perf_func(summary), log=self.log_func(summary))
        name_suffix = "" if (suff := self.name_suffix_func(summary)) == "" else f"_{suff}"
        name = f"{self.last_index}{name_suffix}"
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

    def load(self, idx=-1, map_location=None):
        return self._load_cp_in_path(self.experiment_dir / self.saved[idx],
                                     map_location=map_location)

    def load_last(self, map_location=None):
        return self.load(map_location=map_location)

    def load_best(self, map_location=None):
        return self._load_cp_in_path(self.best_checkpoint_path, map_location=map_location)

    def load_all(self, map_location=None):
        for i in range(len(self.saved)):
            yield self.load(i, map_location=map_location)

    def _load_cp_in_path(self, path, map_location=None):
        self._logger.info(f"Loading checkpoint {path.name}.")
        cp = Checkpoint.load(path, map_location=map_location)
        self.last_index = (pi if isinstance(pi := cp.progress_info, dict) else pi.__dict__)['index']
        self.info = cp.info
        self.resuming_required = False
        return cp.state, cp.summary, self.last_index

    def remove_old_checkpoints(self, n_recent_kept=None, n_best_kept=None):
        n_recent_kept = self.n_recent_kept if n_recent_kept is None else n_recent_kept
        n_best_kept = self.n_best_kept if n_best_kept is None else n_best_kept
        best = set(sorted(self.saved, key=self.id_to_perf.__getitem__, reverse=True)[:n_best_kept])
        olds = [] if math.isinf(max(n_best_kept, n_recent_kept)) else self.saved[:-n_recent_kept]
        to_remove = [x for x in olds if x not in best]
        for p in to_remove:
            path = self.experiment_dir / p
            self.saved.remove(p)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                warnings.warn(f"Old checkpoint {path} is already deleted.")
        self.id_to_perf = {k: self.id_to_perf[k] for k in self.saved}
        if len(self.saved) == 0 and self.experiment_dir.exists():
            shutil.rmtree(self.experiment_dir)

    def __repr__(self):
        return (
            f"CheckpointManager(checkpoints_root={self.checkpoints_root!r}, "
            f"experiment_name={self.experiment_name!r}, "
            f"info={self.info!r}, n_recent_kept={self.n_recent_kept!r}, "
            f"n_best_kept={self.n_best_kept!r}, "
            f"separately_saved_state_parts={self.separately_saved_state_parts!r}, "
            f"resuming_required={self.resuming_required!r})"
        )

    def __str__(self):
        return f"{repr(self)} with checkpoints {repr(self.saved)}"
